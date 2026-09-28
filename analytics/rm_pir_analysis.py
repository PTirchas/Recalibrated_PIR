"""Regenerate the RM-PIR analyses, tables, and figures.

This pipeline implements the manuscript's annual nested rolling protocol from
cached official EuroLeague responses.  For each target season it uses at most
the three most recent completed seasons, selects ``tau`` and ``alpha`` on a
chronological 70/30 split of distinct dates, refits the selected specification
on the complete calibration window, applies the non-PTS RMS scale anchor, and
then locks the coefficients before scoring the target season.

Run from the repository root with::

    python -m analytics.rm_pir_analysis

Inputs are the analysis-ready official-source tables under ``data/processed``.
The script does not make network requests.
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
import sys
from collections import OrderedDict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import check_grad, minimize
from scipy.special import expit
from scipy.stats import binomtest


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data" / "processed"
DERIVED = ROOT / "tables"
TABLES = DERIVED / "latex"
FIGURES = ROOT / "figures"

TEAM_BOXSCORES = DATA / "team_boxscores.csv"
PLAYER_GAMES = DATA / "player_games.csv"
TEAM_NAMES = DATA / "team_names.csv"
SOURCE_EXCLUSIONS = DATA / "source_exclusions.csv"

SEASONS = ("E2020", "E2021", "E2022", "E2023", "E2024", "E2025")
SEASON_LABELS = {
    "E2020": "2020--21",
    "E2021": "2021--22",
    "E2022": "2022--23",
    "E2023": "2023--24",
    "E2024": "2024--25",
    "E2025": "2025--26",
}
EXPECTED_GAMES = {
    "E2020": 328,
    "E2021": 299,
    "E2022": 328,
    "E2023": 331,
    "E2024": 330,
    "E2025": 402,
}
EXPERIMENTS = OrderedDict(
    [
        ("E2021", ("E2020",)),
        ("E2022", ("E2020", "E2021")),
        ("E2023", ("E2020", "E2021", "E2022")),
        ("E2024", ("E2021", "E2022", "E2023")),
        ("E2025", ("E2022", "E2023", "E2024")),
    ]
)

COMPONENTS = ("TR", "AST", "STL", "BLK", "FD", "MFG", "MFT", "TO", "BLKA", "FC")
SIGNS = np.array([1, 1, 1, 1, 1, -1, -1, -1, -1, -1], dtype=float)
GROUPS = ("TR", "AST", "STL", "BLOCK_PAIR", "FOUL_PAIR", "MFG", "MFT", "TO")
EXPANSION = np.zeros((len(COMPONENTS), len(GROUPS)), dtype=float)
EXPANSION[COMPONENTS.index("TR"), GROUPS.index("TR")] = 1
EXPANSION[COMPONENTS.index("AST"), GROUPS.index("AST")] = 1
EXPANSION[COMPONENTS.index("STL"), GROUPS.index("STL")] = 1
EXPANSION[COMPONENTS.index("BLK"), GROUPS.index("BLOCK_PAIR")] = 1
EXPANSION[COMPONENTS.index("BLKA"), GROUPS.index("BLOCK_PAIR")] = 1
EXPANSION[COMPONENTS.index("FD"), GROUPS.index("FOUL_PAIR")] = 1
EXPANSION[COMPONENTS.index("FC"), GROUPS.index("FOUL_PAIR")] = 1
EXPANSION[COMPONENTS.index("MFG"), GROUPS.index("MFG")] = 1
EXPANSION[COMPONENTS.index("MFT"), GROUPS.index("MFT")] = 1
EXPANSION[COMPONENTS.index("TO"), GROUPS.index("TO")] = 1

SOURCE_FIELDS = {
    "PTS": "points",
    "TR": "totalRebounds",
    "AST": "assistances",
    "STL": "steals",
    "BLK": "blocksFavour",
    "FD": "foulsReceived",
    "FGA": "fieldGoalsAttemptedTotal",
    "FGM": "fieldGoalsMadeTotal",
    "FTA": "freeThrowsAttempted",
    "FTM": "freeThrowsMade",
    "OREB": "offensiveRebounds",
    "TO": "turnovers",
    "BLKA": "blocksAgainst",
    "FC": "foulsCommited",
    "OfficialPIR": "valuation",
}

TAU_GRID = (1.0, 2.0, 3.0, 5.0, 10.0)
ALPHA_GRID = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
EXPANDED_TAU_GRID = (0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0)
EXPANDED_ALPHA_GRID = (0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)
LOWER = 0.5
UPPER = 1.5
TOLERANCE = 1e-9
RMS_EPSILON = 1e-12
SEED = 20260730
PAIRED_BOOTSTRAPS = 20_000
COEFFICIENT_BOOTSTRAPS = 500

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
    }
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_metadata() -> pd.DataFrame:
    metadata = pd.read_csv(TEAM_BOXSCORES)
    metadata = metadata[metadata["Season"].isin(SEASONS)].copy()
    metadata["Date"] = pd.to_datetime(metadata["Date"], utc=True)
    keep = [
        "Season",
        "GameCode",
        "Date",
        "Round",
        "Phase",
        "GameMinutes",
        "Side",
        "TeamCode",
        "OpponentCode",
    ]
    metadata = metadata[keep]
    if metadata.duplicated(["Season", "GameCode", "Side"]).any():
        raise RuntimeError("Duplicate team-game metadata rows")
    return metadata


def load_team_names() -> dict[str, str]:
    names = pd.read_csv(TEAM_NAMES, dtype={"TeamCode": str, "TeamName": str})
    if names["TeamCode"].duplicated().any():
        raise RuntimeError("Duplicate team codes in team_names.csv")
    return dict(zip(names["TeamCode"], names["TeamName"]))


def load_team_totals(metadata: pd.DataFrame) -> pd.DataFrame:
    totals = pd.read_csv(TEAM_BOXSCORES)
    totals = totals[totals["Season"].isin(SEASONS)].copy()
    totals["Date"] = pd.to_datetime(totals["Date"], utc=True)
    if totals.duplicated(["Season", "GameCode", "Side"]).any():
        raise RuntimeError("Duplicate official team box-score rows")
    if totals[["Date", "TeamCode", "OpponentCode"]].isna().any().any():
        raise RuntimeError("Unmatched official team totals")
    totals["MFG"] = totals["FGA"] - totals["FGM"]
    totals["MFT"] = totals["FTA"] - totals["FTM"]
    totals["ReconstructedOfficialPIR"] = totals["PTS"] + sum(
        SIGNS[index] * totals[component] for index, component in enumerate(COMPONENTS)
    )
    totals["EstimatedPossessions"] = (
        totals["FGA"] + 0.44 * totals["FTA"] - totals["OREB"] + totals["TO"]
    )
    if not totals.groupby(["Season", "GameCode"]).size().eq(2).all():
        raise RuntimeError("Every included game must contain two team rows")
    if totals.groupby("Season")["GameCode"].nunique().to_dict() != EXPECTED_GAMES:
        raise RuntimeError("Unexpected season game counts")
    if not np.allclose(totals["OfficialPIR"], totals["ReconstructedOfficialPIR"], atol=1e-10):
        raise RuntimeError("Official team PIR reconstruction failed")
    if (totals[["MFG", "MFT"]] < 0).any().any():
        raise RuntimeError("Negative attempts-minus-makes value")
    return totals.sort_values(["Season", "Date", "GameCode", "Side"]).reset_index(drop=True)


def load_players() -> pd.DataFrame:
    players = pd.read_csv(PLAYER_GAMES, dtype={"PlayerCode": str})
    players = players[players["Season"].isin(SEASONS)].copy()
    players["Date"] = pd.to_datetime(players["Date"], utc=True)
    players["MFG"] = players["FGA"] - players["FGM"]
    players["MFT"] = players["FTA"] - players["FTM"]
    reconstructed = players["PTS"] + sum(
        SIGNS[index] * players[component] for index, component in enumerate(COMPONENTS)
    )
    players["ReconstructedOfficialPIR"] = reconstructed
    if not np.allclose(players["PIR"], reconstructed, atol=1e-10):
        raise RuntimeError("Official player PIR reconstruction failed")
    if (players[["MFG", "MFT"]] < 0).any().any():
        raise RuntimeError("Negative player attempts-minus-makes value")
    return players


def build_pairs(totals: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (season, game_code), frame in totals.groupby(["Season", "GameCode"], sort=True):
        if frame["PTS"].nunique() != 2:
            raise RuntimeError(f"Tied or malformed final score: {season}/{game_code}")
        winner = frame.loc[frame["PTS"].idxmax()]
        loser = frame.loc[frame["PTS"].idxmin()]
        row: dict[str, object] = {
            "Season": season,
            "GameCode": int(game_code),
            "Date": winner["Date"],
            "Round": winner["Round"],
            "Phase": winner["Phase"],
            "GameMinutes": float(winner["GameMinutes"]),
            "GamePossessions": float(frame["EstimatedPossessions"].mean()),
            "Winner": winner["TeamCode"],
            "Loser": loser["TeamCode"],
            "DeltaPTS": float(winner["PTS"] - loser["PTS"]),
        }
        for component in COMPONENTS:
            row[f"Delta{component}"] = float(winner[component] - loser[component])
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["Season", "Date", "GameCode"]).reset_index(drop=True)


def design_matrix(pairs: pd.DataFrame, pace_normalized: bool = False) -> np.ndarray:
    matrix = pairs[[f"Delta{component}" for component in COMPONENTS]].to_numpy(float) * SIGNS
    if pace_normalized:
        scale = 70.0 / pairs["GamePossessions"].to_numpy(float)
        matrix = matrix * scale[:, None]
    return matrix


def classification(values: Iterable[float], tolerance: float = TOLERANCE) -> tuple[int, int, int]:
    values = np.asarray(values, dtype=float)
    result = (
        int((values > tolerance).sum()),
        int((np.abs(values) <= tolerance).sum()),
        int((values < -tolerance).sum()),
    )
    if sum(result) != len(values):
        raise RuntimeError("Ordering classification did not partition observations")
    return result


def fit_matrix(
    grouped: np.ndarray,
    expansion: np.ndarray,
    alpha: float,
    tau: float,
    lower: float = LOWER,
    upper: float = UPPER,
    start: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    grouped = np.asarray(grouped, dtype=float)
    expansion = np.asarray(expansion, dtype=float)
    if start is None:
        start = np.ones(grouped.shape[1], dtype=float)

    def objective(group_weights: np.ndarray) -> float:
        contrast = grouped @ group_weights
        soft_loss = np.mean(np.logaddexp(0.0, -contrast / tau))
        expanded = expansion @ group_weights
        ridge = alpha * np.mean(np.square(expanded - 1.0))
        return float(soft_loss + ridge)

    def gradient(group_weights: np.ndarray) -> np.ndarray:
        contrast = grouped @ group_weights
        loss_gradient = -(grouped.T @ expit(-contrast / tau)) / (len(grouped) * tau)
        expanded = expansion @ group_weights
        ridge_gradient = 2.0 * alpha * expansion.T @ (expanded - 1.0) / len(COMPONENTS)
        return loss_gradient + ridge_gradient

    result = minimize(
        objective,
        np.clip(np.asarray(start, dtype=float), lower, upper),
        jac=gradient,
        method="L-BFGS-B",
        bounds=[(lower, upper)] * grouped.shape[1],
        options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 5000, "maxls": 50},
    )
    fallback = False
    if not result.success:
        fallback = True
        result = minimize(
            objective,
            np.clip(result.x, lower, upper),
            jac=gradient,
            method="SLSQP",
            bounds=[(lower, upper)] * grouped.shape[1],
            options={"ftol": 1e-11, "maxiter": 5000},
        )
    if not result.success:
        raise RuntimeError(f"Coefficient optimization failed: {result.message}")
    fitted = np.asarray(result.x, dtype=float)
    audit = {
        "Objective": float(result.fun),
        "Converged": bool(result.success),
        "Iterations": int(getattr(result, "nit", -1)),
        "Fallback": fallback,
        "Message": str(result.message),
        "BoundaryCount": int(
            (np.isclose(fitted, lower, atol=1e-7) | np.isclose(fitted, upper, atol=1e-7)).sum()
        ),
    }
    return fitted, audit


def fit_pairs(
    pairs: pd.DataFrame,
    alpha: float,
    tau: float,
    lower: float = LOWER,
    upper: float = UPPER,
    expansion: np.ndarray = EXPANSION,
    pace_normalized: bool = False,
    start: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    grouped = design_matrix(pairs, pace_normalized=pace_normalized) @ expansion
    return fit_matrix(grouped, expansion, alpha, tau, lower, upper, start)


def select_hyperparameters(
    calibration: pd.DataFrame,
    expansion: np.ndarray = EXPANSION,
    pace_normalized: bool = False,
    tau_grid: Iterable[float] = TAU_GRID,
    alpha_grid: Iterable[float] = ALPHA_GRID,
    split_fraction: float = 0.70,
) -> tuple[float, float, pd.Timestamp, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ordered = calibration.sort_values(["Date", "GameCode"]).copy()
    distinct_dates = np.array(sorted(ordered["Date"].dt.normalize().unique()))
    split_index = int(np.floor(split_fraction * len(distinct_dates)))
    if split_index <= 0 or split_index >= len(distinct_dates):
        raise RuntimeError("Chronological split is not estimable")
    training_dates = set(distinct_dates[:split_index])
    validation_dates = set(distinct_dates[split_index:])
    normalized_dates = ordered["Date"].dt.normalize()
    training = ordered[normalized_dates.isin(training_dates)].copy()
    validation = ordered[normalized_dates.isin(validation_dates)].copy()
    cutoff = pd.Timestamp(distinct_dates[split_index])

    training_grouped = design_matrix(training, pace_normalized) @ expansion
    validation_grouped = design_matrix(validation, pace_normalized) @ expansion
    rows: list[dict[str, object]] = []
    for tau in tau_grid:
        for alpha in alpha_grid:
            raw, audit = fit_matrix(training_grouped, expansion, alpha, tau)
            validation_contrast = validation_grouped @ raw
            validation_counts = classification(validation_contrast)
            rows.append(
                {
                    "Tau": tau,
                    "Alpha": alpha,
                    "TrainingGames": len(training),
                    "ValidationGames": len(validation),
                    "CutoffDate": cutoff.date().isoformat(),
                    "SplitFraction": split_fraction,
                    "ValidationWinnerHigher": validation_counts[0],
                    "ValidationTies": validation_counts[1],
                    "ValidationLoserHigher": validation_counts[2],
                    "ValidationSoftLoss": float(
                        np.mean(np.logaddexp(0.0, -validation_contrast / tau))
                    ),
                    **audit,
                }
            )
    grid = pd.DataFrame(rows)
    # The declared criterion is strict non-PTS concordance.  Count ties are
    # broken by stronger regularization, smaller tau, and then lower soft loss.
    selected = grid.sort_values(
        ["ValidationWinnerHigher", "Alpha", "Tau", "ValidationSoftLoss"],
        ascending=[False, False, True, True],
    ).iloc[0]
    return (
        float(selected["Tau"]),
        float(selected["Alpha"]),
        cutoff,
        grid,
        training,
        validation,
    )


def select_hyperparameters_multicutoff(
    calibration: pd.DataFrame,
    expansion: np.ndarray = EXPANSION,
    pace_normalized: bool = False,
    tau_grid: Iterable[float] = TAU_GRID,
    alpha_grid: Iterable[float] = ALPHA_GRID,
) -> tuple[float, float, pd.DataFrame]:
    """Select hyperparameters across three forward-chaining validation blocks."""

    ordered = calibration.sort_values(["Date", "GameCode"]).copy()
    distinct_dates = np.array(sorted(ordered["Date"].dt.normalize().unique()))
    normalized_dates = ordered["Date"].dt.normalize()
    folds = ((0.50, 0.65), (0.65, 0.80), (0.80, 0.95))
    rows: list[dict[str, object]] = []
    for tau in tau_grid:
        for alpha in alpha_grid:
            fold_successes = 0
            fold_games = 0
            weighted_loss = 0.0
            converged = True
            boundary_count = 0
            fold_descriptions: list[str] = []
            for train_fraction, validation_end_fraction in folds:
                train_end = int(np.floor(train_fraction * len(distinct_dates)))
                validation_end = int(np.floor(validation_end_fraction * len(distinct_dates)))
                training_dates = set(distinct_dates[:train_end])
                validation_dates = set(distinct_dates[train_end:validation_end])
                training = ordered[normalized_dates.isin(training_dates)]
                validation = ordered[normalized_dates.isin(validation_dates)]
                if training.empty or validation.empty:
                    raise RuntimeError("Multi-cutoff chronological fold is not estimable")
                training_grouped = design_matrix(training, pace_normalized) @ expansion
                validation_grouped = design_matrix(validation, pace_normalized) @ expansion
                raw, audit = fit_matrix(training_grouped, expansion, alpha, tau)
                validation_contrast = validation_grouped @ raw
                successes = classification(validation_contrast)[0]
                loss = float(np.mean(np.logaddexp(0.0, -validation_contrast / tau)))
                fold_successes += successes
                fold_games += len(validation)
                weighted_loss += len(validation) * loss
                converged = converged and bool(audit["Converged"])
                boundary_count += int(audit["BoundaryCount"])
                fold_descriptions.append(
                    f"{pd.Timestamp(distinct_dates[train_end]).date().isoformat()}:"
                    f"{len(training)}/{len(validation)}"
                )
            rows.append(
                {
                    "Tau": tau,
                    "Alpha": alpha,
                    "ValidationWinnerHigher": fold_successes,
                    "ValidationGames": fold_games,
                    "ValidationSoftLoss": weighted_loss / fold_games,
                    "Converged": converged,
                    "BoundaryCountAcrossFolds": boundary_count,
                    "Folds": ";".join(fold_descriptions),
                }
            )
    grid = pd.DataFrame(rows)
    selected = grid.sort_values(
        ["ValidationWinnerHigher", "Alpha", "Tau", "ValidationSoftLoss"],
        ascending=[False, False, True, True],
    ).iloc[0]
    return float(selected["Tau"]), float(selected["Alpha"]), grid


def rms_anchor(
    calibration: pd.DataFrame,
    raw_magnitudes: np.ndarray,
    pace_normalized: bool = False,
) -> tuple[float, dict[str, float]]:
    matrix = design_matrix(calibration, pace_normalized)
    official = matrix @ np.ones(len(COMPONENTS))
    learned = matrix @ np.asarray(raw_magnitudes, dtype=float)
    official_rms = float(np.sqrt(np.mean(np.square(official))))
    learned_rms = float(np.sqrt(np.mean(np.square(learned))))
    if learned_rms <= RMS_EPSILON:
        raise RuntimeError("Learned non-PTS RMS is below the denominator safeguard")
    gamma = official_rms / learned_rms
    scaled = gamma * learned
    diagnostic = {
        "OfficialNonpointMean": float(official.mean()),
        "RawNonpointMean": float(learned.mean()),
        "FinalNonpointMean": float(scaled.mean()),
        "OfficialNonpointSD": float(official.std(ddof=0)),
        "RawNonpointSD": float(learned.std(ddof=0)),
        "FinalNonpointSD": float(scaled.std(ddof=0)),
        "OfficialNonpointRMS": official_rms,
        "RawNonpointRMS": learned_rms,
        "FinalNonpointRMS": float(np.sqrt(np.mean(np.square(scaled)))),
        "RMSIdentityError": float(abs(official_rms - np.sqrt(np.mean(np.square(scaled))))),
    }
    return float(gamma), diagnostic


def alternative_scale_anchor(
    calibration: pd.DataFrame,
    raw_magnitudes: np.ndarray,
    method: str,
) -> float:
    """Return a sensitivity-only scale factor under an alternative anchor."""

    matrix = design_matrix(calibration)
    official = matrix @ np.ones(len(COMPONENTS))
    learned = matrix @ np.asarray(raw_magnitudes, dtype=float)
    if method == "Centered SD":
        numerator = float(official.std(ddof=0))
        denominator = float(learned.std(ddof=0))
    elif method == "Mean absolute contrast":
        numerator = float(np.mean(np.abs(official)))
        denominator = float(np.mean(np.abs(learned)))
    else:
        raise ValueError(f"Unknown scale anchor: {method}")
    if denominator <= RMS_EPSILON:
        raise RuntimeError(f"{method} denominator is below the safeguard")
    return numerator / denominator


def wilson_interval(failures: int, games: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson 95% interval for a binomial ordering-error proportion, in percent."""

    if games <= 0:
        raise ValueError("A Wilson interval requires at least one game")
    proportion = failures / games
    denominator = 1.0 + z**2 / games
    center = (proportion + z**2 / (2.0 * games)) / denominator
    half_width = z * np.sqrt(
        proportion * (1.0 - proportion) / games + z**2 / (4.0 * games**2)
    ) / denominator
    return 100.0 * (center - half_width), 100.0 * (center + half_width)


def symmetric_relative_error(previous: float, new: float) -> float:
    denominator = abs(previous) + abs(new)
    return 0.0 if denominator == 0 else 200.0 * abs(previous - new) / denominator


def paired_bootstrap_interval(official_failure: np.ndarray, vp_failure: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    change = 100.0 * (vp_failure.astype(float) - official_failure.astype(float))
    draws = np.empty(PAIRED_BOOTSTRAPS, dtype=float)
    for start in range(0, PAIRED_BOOTSTRAPS, 1000):
        size = min(1000, PAIRED_BOOTSTRAPS - start)
        indices = rng.integers(0, len(change), size=(size, len(change)))
        draws[start : start + size] = change[indices].mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def evaluate_values(
    frame: pd.DataFrame,
    official: np.ndarray,
    proposed: np.ndarray,
    estimand: str,
    experiment: str,
    calibration_label: str,
    tau: float,
    alpha: float,
    seed: int,
) -> tuple[dict[str, object], dict[str, object]]:
    official_counts = classification(official)
    proposed_counts = classification(proposed)
    games = len(frame)
    official_failure = official <= TOLERANCE
    proposed_failure = proposed <= TOLERANCE
    official_error = 100.0 * official_failure.mean()
    proposed_error = 100.0 * proposed_failure.mean()
    official_ci_low, official_ci_high = wilson_interval(int(official_failure.sum()), games)
    proposed_ci_low, proposed_ci_high = wilson_interval(int(proposed_failure.sum()), games)
    corrections = int((official_failure & ~proposed_failure).sum())
    new_failures = int((~official_failure & proposed_failure).sum())
    both_success = int((~official_failure & ~proposed_failure).sum())
    both_failure = int((official_failure & proposed_failure).sum())
    ci_low, ci_high = paired_bootstrap_interval(official_failure, proposed_failure, seed)
    discordant = corrections + new_failures
    mcnemar_p = 1.0 if discordant == 0 else float(
        binomtest(min(corrections, new_failures), discordant, 0.5, alternative="two-sided").pvalue
    )
    summary = {
        "Experiment": experiment,
        "TargetSeason": str(frame["Season"].iloc[0]) if frame["Season"].nunique() == 1 else "Pooled",
        "CalibrationSeasons": calibration_label,
        "Estimand": estimand,
        "Games": games,
        "OfficialWinnerHigher": official_counts[0],
        "OfficialTies": official_counts[1],
        "OfficialLoserHigher": official_counts[2],
        "OfficialErrorRatePercent": official_error,
        "OfficialErrorRateCI2_5Percent": official_ci_low,
        "OfficialErrorRateCI97_5Percent": official_ci_high,
        "VPWinnerHigher": proposed_counts[0],
        "VPTies": proposed_counts[1],
        "VPLoserHigher": proposed_counts[2],
        "VPErrorRatePercent": proposed_error,
        "VPErrorRateCI2_5Percent": proposed_ci_low,
        "VPErrorRateCI97_5Percent": proposed_ci_high,
        "ErrorRateDifferencePP": proposed_error - official_error,
        "ErrorRateDifferenceCI2_5PP": ci_low,
        "ErrorRateDifferenceCI97_5PP": ci_high,
        "SymmetricRelativeErrorPercent": symmetric_relative_error(official_error, proposed_error),
        "Corrections": corrections,
        "NewFailures": new_failures,
        "McNemarExactP": mcnemar_p,
        "Tau": tau,
        "Alpha": alpha,
    }
    transitions = {
        "Experiment": experiment,
        "TargetSeason": summary["TargetSeason"],
        "Estimand": estimand,
        "BothSuccess": both_success,
        "OfficialOnlySuccess": new_failures,
        "VPOnlySuccess": corrections,
        "BothFailure": both_failure,
        "McNemarExactP": mcnemar_p,
    }
    return summary, transitions


def rolling_analysis(
    totals: pd.DataFrame, pairs: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summaries: list[dict[str, object]] = []
    transitions: list[dict[str, object]] = []
    model_rows: list[dict[str, object]] = []
    coefficient_rows: list[dict[str, object]] = []
    game_frames: list[pd.DataFrame] = []
    tuning_frames: list[pd.DataFrame] = []

    for experiment_number, (target, calibration_seasons) in enumerate(EXPERIMENTS.items(), start=1):
        if target in calibration_seasons or any(season >= target for season in calibration_seasons):
            raise RuntimeError(f"Temporal leakage in {target}")
        calibration = pairs[pairs["Season"].isin(calibration_seasons)].copy()
        target_pairs = pairs[pairs["Season"] == target].copy()
        tau, alpha, cutoff, grid, training, validation = select_hyperparameters(calibration)
        raw_group, audit = fit_pairs(calibration, alpha, tau)
        raw_magnitudes = EXPANSION @ raw_group
        gamma, scale_diagnostic = rms_anchor(calibration, raw_magnitudes)
        final_magnitudes = gamma * raw_magnitudes
        signed_coefficients = SIGNS * final_magnitudes
        experiment = f"X{experiment_number}"
        calibration_label = "+".join(calibration_seasons)

        tuning = grid.copy()
        tuning.insert(0, "Experiment", experiment)
        tuning.insert(1, "TargetSeason", target)
        tuning["Selected"] = (tuning["Tau"] == tau) & (tuning["Alpha"] == alpha)
        tuning_frames.append(tuning)

        model_rows.append(
            {
                "Experiment": experiment,
                "TargetSeason": target,
                "TargetSeasonLabel": SEASON_LABELS[target],
                "CalibrationSeasons": calibration_label,
                "CalibrationStart": calibration["Date"].min().date().isoformat(),
                "CalibrationEnd": calibration["Date"].max().date().isoformat(),
                "CalibrationGames": len(calibration),
                "TargetStartDate": target_pairs["Date"].min().date().isoformat(),
                "InnerCutoffDate": cutoff.date().isoformat(),
                "InnerTrainingGames": len(training),
                "InnerValidationGames": len(validation),
                "Tau": tau,
                "Alpha": alpha,
                "LowerBound": LOWER,
                "UpperBound": UPPER,
                "Gamma": gamma,
                "RawMinimum": float(raw_magnitudes.min()),
                "RawMaximum": float(raw_magnitudes.max()),
                "FinalMagnitudeMinimum": float(final_magnitudes.min()),
                "FinalMagnitudeMaximum": float(final_magnitudes.max()),
                **scale_diagnostic,
                **audit,
            }
        )
        for index, component in enumerate(COMPONENTS):
            coefficient_rows.append(
                {
                    "Experiment": experiment,
                    "TargetSeason": target,
                    "Component": component,
                    "Sign": int(SIGNS[index]),
                    "RawMagnitude": raw_magnitudes[index],
                    "Gamma": gamma,
                    "FinalMagnitude": final_magnitudes[index],
                    "SignedCoefficient": signed_coefficients[index],
                    "RawAtLowerBound": bool(np.isclose(raw_magnitudes[index], LOWER, atol=1e-7)),
                    "RawAtUpperBound": bool(np.isclose(raw_magnitudes[index], UPPER, atol=1e-7)),
                }
            )

        work = target_pairs.copy()
        matrix = design_matrix(work)
        work["Experiment"] = experiment
        work["CalibrationSeasons"] = calibration_label
        work["OfficialNonpointDiff"] = matrix @ np.ones(len(COMPONENTS))
        work["VPNonpointDiff"] = matrix @ final_magnitudes
        work["OfficialFullDiff"] = work["DeltaPTS"] + work["OfficialNonpointDiff"]
        work["VPFullDiff"] = work["DeltaPTS"] + work["VPNonpointDiff"]
        game_frames.append(work)
        for offset, (estimand, official_column, proposed_column) in enumerate(
            (
                ("Non-PTS", "OfficialNonpointDiff", "VPNonpointDiff"),
                ("Full", "OfficialFullDiff", "VPFullDiff"),
            )
        ):
            summary, transition = evaluate_values(
                work,
                work[official_column].to_numpy(float),
                work[proposed_column].to_numpy(float),
                estimand,
                experiment,
                calibration_label,
                tau,
                alpha,
                SEED + 100 * experiment_number + offset,
            )
            summaries.append(summary)
            transitions.append(transition)

    games = pd.concat(game_frames, ignore_index=True)
    summary = pd.DataFrame(summaries)
    transition_table = pd.DataFrame(transitions)
    for offset, (estimand, official_column, proposed_column) in enumerate(
        (
            ("Non-PTS", "OfficialNonpointDiff", "VPNonpointDiff"),
            ("Full", "OfficialFullDiff", "VPFullDiff"),
        )
    ):
        pooled_summary, pooled_transition = evaluate_values(
            games,
            games[official_column].to_numpy(float),
            games[proposed_column].to_numpy(float),
            estimand,
            "Pooled",
            "target-specific",
            np.nan,
            np.nan,
            SEED + 900 + offset,
        )
        summary = pd.concat([summary, pd.DataFrame([pooled_summary])], ignore_index=True)
        transition_table = pd.concat(
            [transition_table, pd.DataFrame([pooled_transition])], ignore_index=True
        )
    models = pd.DataFrame(model_rows)
    coefficients = pd.DataFrame(coefficient_rows)
    tuning = pd.concat(tuning_frames, ignore_index=True)
    return summary, transition_table, models, coefficients, games, tuning


def coefficient_bootstrap(
    pairs: pd.DataFrame, models: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED + 4000)
    rows: list[dict[str, object]] = []
    performance_rows: list[dict[str, object]] = []
    for model in models.itertuples(index=False):
        calibration_seasons = str(model.CalibrationSeasons).split("+")
        calibration = pairs[pairs["Season"].isin(calibration_seasons)].reset_index(drop=True)
        target = pairs[pairs["Season"] == model.TargetSeason].reset_index(drop=True)
        target_matrix = design_matrix(target)
        for repetition in range(COEFFICIENT_BOOTSTRAPS):
            sample = calibration.iloc[rng.integers(0, len(calibration), len(calibration))]
            raw_group, audit = fit_pairs(sample, float(model.Alpha), float(model.Tau))
            raw = EXPANSION @ raw_group
            gamma, _ = rms_anchor(sample, raw)
            final = gamma * raw
            for index, component in enumerate(COMPONENTS):
                rows.append(
                    {
                        "TargetSeason": model.TargetSeason,
                        "Repetition": repetition,
                        "Component": component,
                        "RawMagnitude": raw[index],
                        "Gamma": gamma,
                        "SignedCoefficient": SIGNS[index] * final[index],
                        "RawAtLowerBound": bool(np.isclose(raw[index], LOWER, atol=1e-7)),
                        "RawAtUpperBound": bool(np.isclose(raw[index], UPPER, atol=1e-7)),
                        "Converged": audit["Converged"],
                    }
                )
            proposed_nonpoint = target_matrix @ final
            proposed_full = target["DeltaPTS"].to_numpy(float) + proposed_nonpoint
            for estimand, values in (("Non-PTS", proposed_nonpoint), ("Full", proposed_full)):
                counts = classification(values)
                errors = counts[1] + counts[2]
                performance_rows.append(
                    {
                        "TargetSeason": model.TargetSeason,
                        "Repetition": repetition,
                        "Estimand": estimand,
                        "VPWinnerHigher": counts[0],
                        "VPErrors": errors,
                        "VPErrorRatePercent": 100.0 * errors / len(target),
                    }
                )
    draws = pd.DataFrame(rows)
    summary = (
        draws.groupby(["TargetSeason", "Component"], sort=False)
        .agg(
            RawCI2_5=("RawMagnitude", lambda x: np.quantile(x, 0.025)),
            RawMedian=("RawMagnitude", "median"),
            RawCI97_5=("RawMagnitude", lambda x: np.quantile(x, 0.975)),
            SignedCI2_5=("SignedCoefficient", lambda x: np.quantile(x, 0.025)),
            SignedMedian=("SignedCoefficient", "median"),
            SignedCI97_5=("SignedCoefficient", lambda x: np.quantile(x, 0.975)),
            GammaCI2_5=("Gamma", lambda x: np.quantile(x, 0.025)),
            GammaMedian=("Gamma", "median"),
            GammaCI97_5=("Gamma", lambda x: np.quantile(x, 0.975)),
            RawLowerBoundaryFrequencyPercent=("RawAtLowerBound", lambda x: 100.0 * np.mean(x)),
            RawUpperBoundaryFrequencyPercent=("RawAtUpperBound", lambda x: 100.0 * np.mean(x)),
            Converged=("Converged", "all"),
        )
        .reset_index()
    )
    performance = pd.DataFrame(performance_rows)
    performance_summary = (
        performance.groupby(["TargetSeason", "Estimand"], sort=False)
        .agg(
            VPWinnerHigherCI2_5=("VPWinnerHigher", lambda x: np.quantile(x, 0.025)),
            VPWinnerHigherMedian=("VPWinnerHigher", "median"),
            VPWinnerHigherCI97_5=("VPWinnerHigher", lambda x: np.quantile(x, 0.975)),
            VPErrorsCI2_5=("VPErrors", lambda x: np.quantile(x, 0.025)),
            VPErrorsMedian=("VPErrors", "median"),
            VPErrorsCI97_5=("VPErrors", lambda x: np.quantile(x, 0.975)),
            VPErrorRateCI2_5Percent=("VPErrorRatePercent", lambda x: np.quantile(x, 0.025)),
            VPErrorRateMedianPercent=("VPErrorRatePercent", "median"),
            VPErrorRateCI97_5Percent=("VPErrorRatePercent", lambda x: np.quantile(x, 0.975)),
        )
        .reset_index()
    )
    return summary, performance_summary


def player_and_team_outputs(
    totals: pd.DataFrame,
    players: pd.DataFrame,
    coefficients: pd.DataFrame,
    names: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    e2025_coefficients = coefficients[coefficients["TargetSeason"] == "E2025"].set_index("Component")
    magnitudes = e2025_coefficients.loc[list(COMPONENTS), "FinalMagnitude"].to_numpy(float)

    e2025_players = players[(players["Season"] == "E2025") & (players["Minutes"] > 0)].copy()
    player_matrix = e2025_players[list(COMPONENTS)].to_numpy(float) * SIGNS
    e2025_players["VPPIR"] = e2025_players["PTS"].to_numpy(float) + player_matrix @ magnitudes
    grouped = e2025_players.groupby(["PlayerCode", "Player"], sort=False)
    player_season = grouped.agg(
        GP=("GameCode", "nunique"),
        Minutes=("Minutes", "sum"),
        OfficialPIR=("PIR", "sum"),
        VPPIR=("VPPIR", "sum"),
    ).reset_index()
    teams = grouped["TeamCode"].agg(lambda values: "; ".join(sorted(set(values)))).reset_index(name="Teams")
    player_season = player_season.merge(teams, on=["PlayerCode", "Player"], validate="one_to_one")
    player_season["OfficialRank"] = player_season["OfficialPIR"].rank(method="min", ascending=False).astype(int)
    player_season["VPRank"] = player_season["VPPIR"].rank(method="min", ascending=False).astype(int)
    player_season["RankChange"] = player_season["OfficialRank"] - player_season["VPRank"]
    player_season = player_season.sort_values(
        ["VPRank", "OfficialRank", "Player"], ascending=[True, True, True]
    ).reset_index(drop=True)

    e2025_teams = totals[totals["Season"] == "E2025"].copy()
    team_matrix = e2025_teams[list(COMPONENTS)].to_numpy(float) * SIGNS
    e2025_teams["VPPIR"] = e2025_teams["PTS"].to_numpy(float) + team_matrix @ magnitudes

    def aggregate_teams(frame: pd.DataFrame, universe: str) -> pd.DataFrame:
        table = frame.groupby("TeamCode", sort=False).agg(
            GP=("GameCode", "nunique"),
            OfficialPIRTotal=("OfficialPIR", "sum"),
            VPPIRTotal=("VPPIR", "sum"),
            OfficialPIRPerGame=("OfficialPIR", "mean"),
            VPPIRPerGame=("VPPIR", "mean"),
        ).reset_index()
        table["Team"] = table["TeamCode"].map(names).fillna(table["TeamCode"])
        table["Universe"] = universe
        table["OfficialTotalRank"] = table["OfficialPIRTotal"].rank(
            method="min", ascending=False
        ).astype(int)
        table["VPTotalRank"] = table["VPPIRTotal"].rank(
            method="min", ascending=False
        ).astype(int)
        table["OfficialPerGameRank"] = table["OfficialPIRPerGame"].rank(
            method="min", ascending=False
        ).astype(int)
        table["VPPerGameRank"] = table["VPPIRPerGame"].rank(
            method="min", ascending=False
        ).astype(int)
        table["TotalRankChange"] = table["OfficialTotalRank"] - table["VPTotalRank"]
        table["PerGameRankChange"] = (
            table["OfficialPerGameRank"] - table["VPPerGameRank"]
        )
        # The generic rank fields retain the equal-exposure/per-game convention used
        # by the primary regular-season analysis and its rank-movement figure.
        table["OfficialRank"] = table["OfficialPerGameRank"]
        table["VPRank"] = table["VPPerGameRank"]
        table["RankChange"] = table["PerGameRankChange"]
        return table.sort_values(["VPRank", "OfficialRank", "Team"]).reset_index(drop=True)

    regular = aggregate_teams(
        e2025_teams[e2025_teams["Phase"] == "Regular Season"], "Regular season"
    )
    all_stage = aggregate_teams(e2025_teams, "All completed stages")
    if not regular["GP"].eq(38).all():
        raise RuntimeError("The E2025 regular-season team ranking does not have a common 38-game sample")
    return player_season, regular, all_stage


def residual_audit(
    totals: pd.DataFrame,
    players: pd.DataFrame,
    coefficients: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fields = ("PTS", *COMPONENTS)
    sums = (
        players.groupby(["Season", "GameCode", "TeamCode"], as_index=False)[list(fields)]
        .sum()
        .rename(columns={field: f"Player{field}" for field in fields})
    )
    audit = totals.merge(sums, on=["Season", "GameCode", "TeamCode"], how="left", validate="one_to_one")
    if audit[[f"Player{field}" for field in fields]].isna().any().any():
        raise RuntimeError("A team-game could not be matched to player rows")
    for field in fields:
        audit[f"Residual{field}"] = audit[field] - audit[f"Player{field}"]

    component_rows: list[dict[str, object]] = []
    for season, frame in audit.groupby("Season", sort=True):
        for field in fields:
            values = frame[f"Residual{field}"].to_numpy(float)
            component_rows.append(
                {
                    "Season": season,
                    "Component": field,
                    "TeamGames": len(frame),
                    "NonzeroTeamGames": int((np.abs(values) > TOLERANCE).sum()),
                    "NonzeroPercent": 100.0 * float((np.abs(values) > TOLERANCE).mean()),
                    "MeanResidual": float(values.mean()),
                    "MinimumResidual": float(values.min()),
                    "MaximumResidual": float(values.max()),
                    "MaximumAbsoluteResidual": float(np.abs(values).max()),
                }
            )

    identity_rows: list[dict[str, object]] = []
    coefficient_lookup = {
        season: frame.set_index("Component").loc[list(COMPONENTS), "SignedCoefficient"].to_numpy(float)
        for season, frame in coefficients.groupby("TargetSeason")
    }
    for season, frame in audit.groupby("Season", sort=True):
        official_residual = frame["ResidualPTS"].to_numpy(float) + (
            frame[[f"Residual{component}" for component in COMPONENTS]].to_numpy(float) @ SIGNS
        )
        row: dict[str, object] = {
            "Season": season,
            "TeamGames": len(frame),
            "OfficialMaximumAbsoluteWeightedResidual": float(np.abs(official_residual).max()),
            "PTSResidualNonzeroTeamGames": int((np.abs(frame["ResidualPTS"]) > TOLERANCE).sum()),
        }
        if season in coefficient_lookup:
            signed = coefficient_lookup[season]
            vp_residual = frame["ResidualPTS"].to_numpy(float) + (
                frame[[f"Residual{component}" for component in COMPONENTS]].to_numpy(float) @ signed
            )
            team_score = frame["PTS"].to_numpy(float) + (
                frame[list(COMPONENTS)].to_numpy(float) @ signed
            )
            player_score = frame["PlayerPTS"].to_numpy(float) + (
                frame[[f"Player{component}" for component in COMPONENTS]].to_numpy(float) @ signed
            )
            identity_error = team_score - player_score - vp_residual
            row.update(
                {
                    "VPMaximumAbsoluteWeightedResidual": float(np.abs(vp_residual).max()),
                    "VPAdditivityIdentityMaximumAbsoluteError": float(np.abs(identity_error).max()),
                }
            )
        identity_rows.append(row)
    return pd.DataFrame(component_rows), pd.DataFrame(identity_rows)


def build_player_sum_pairs(totals: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    """Rebuild winner--loser contrasts after removing every team-only component residual."""

    fields = ("PTS", *COMPONENTS)
    sums = players.groupby(["Season", "GameCode", "TeamCode"], as_index=False)[list(fields)].sum()
    renamed = sums.rename(columns={field: f"Player{field}" for field in fields})
    player_totals = totals.merge(
        renamed,
        on=["Season", "GameCode", "TeamCode"],
        how="left",
        validate="one_to_one",
    )
    if player_totals[[f"Player{field}" for field in fields]].isna().any().any():
        raise RuntimeError("Player-summed sensitivity could not match a team-game")
    for field in fields:
        player_totals[field] = player_totals[f"Player{field}"]
    return build_pairs(player_totals)


def group_restriction_diagnostics(pairs: pd.DataFrame) -> pd.DataFrame:
    """Report empirical support for the two paired raw-magnitude restrictions."""

    rows: list[dict[str, object]] = []
    for target, calibration_seasons in EXPERIMENTS.items():
        calibration = pairs[pairs["Season"].isin(calibration_seasons)]
        matrix = design_matrix(calibration)
        for first, second in (("BLK", "BLKA"), ("FD", "FC")):
            first_index = COMPONENTS.index(first)
            second_index = COMPONENTS.index(second)
            rows.append(
                {
                    "TargetSeason": target,
                    "CalibrationSeasons": "+".join(calibration_seasons),
                    "Pair": f"{first}-{second}",
                    "Games": len(calibration),
                    "SignedContrastPearsonR": float(
                        np.corrcoef(matrix[:, first_index], matrix[:, second_index])[0, 1]
                    ),
                }
            )
    return pd.DataFrame(rows)


def team_dependence_sensitivity(games: pd.DataFrame) -> pd.DataFrame:
    """Assess E2025 paired effects after omitting every club in turn."""

    target = games[games["Season"] == "E2025"].copy()
    teams = sorted(set(target["Winner"]) | set(target["Loser"]))
    rows: list[dict[str, object]] = []
    for estimand, official_column, proposed_column in (
        ("Non-PTS", "OfficialNonpointDiff", "VPNonpointDiff"),
        ("Full", "OfficialFullDiff", "VPFullDiff"),
    ):
        for omitted in ("None", *teams):
            frame = target if omitted == "None" else target[
                (target["Winner"] != omitted) & (target["Loser"] != omitted)
            ]
            official_failure = frame[official_column].to_numpy(float) <= TOLERANCE
            proposed_failure = frame[proposed_column].to_numpy(float) <= TOLERANCE
            rows.append(
                {
                    "Estimand": estimand,
                    "OmittedTeam": omitted,
                    "Games": len(frame),
                    "OfficialErrors": int(official_failure.sum()),
                    "VPErrors": int(proposed_failure.sum()),
                    "ErrorDifferencePP": 100.0
                    * (proposed_failure.mean() - official_failure.mean()),
                }
            )
    return pd.DataFrame(rows)


def data_and_comparability_audit(totals: pd.DataFrame, pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    audit_rows: list[dict[str, object]] = []
    for season, frame in totals.groupby("Season", sort=True):
        games = frame["GameCode"].nunique()
        pair_frame = pairs[pairs["Season"] == season]
        audit_rows.append(
            {
                "Season": season,
                "IncludedGames": games,
                "TeamRows": len(frame),
                "RegularSeasonGames": frame.loc[frame["Phase"] == "Regular Season", "GameCode"].nunique(),
                "PlayInGames": frame.loc[frame["Phase"] == "Play-In", "GameCode"].nunique(),
                "PlayoffGames": frame.loc[frame["Phase"] == "Playoffs", "GameCode"].nunique(),
                "FinalFourGames": frame.loc[frame["Phase"] == "Final Four", "GameCode"].nunique(),
                "OvertimeGames": int((pair_frame["GameMinutes"] > 40).sum()),
                "MissingPrimaryFields": int(frame[["PTS", *COMPONENTS]].isna().sum().sum()),
                "OfficialPIRReconstructionMismatches": int(
                    (~np.isclose(frame["OfficialPIR"], frame["ReconstructedOfficialPIR"], atol=1e-10)).sum()
                ),
                "NegativeMissedShotRows": int((frame[["MFG", "MFT"]] < 0).any(axis=1).sum()),
                "TiedFinalScores": int((pair_frame["DeltaPTS"] <= 0).sum()),
            }
        )

    distribution_rows: list[dict[str, object]] = []
    previous: dict[str, tuple[float, float]] = {}
    for season in SEASONS:
        frame = totals[totals["Season"] == season]
        for component in ("PTS", *COMPONENTS):
            values = frame[component].to_numpy(float)
            mean = float(values.mean())
            sd = float(values.std(ddof=0))
            standardized_shift = np.nan
            if component in previous:
                prior_mean, prior_sd = previous[component]
                pooled_sd = np.sqrt((prior_sd**2 + sd**2) / 2.0)
                standardized_shift = 0.0 if pooled_sd == 0 else abs(mean - prior_mean) / pooled_sd
            distribution_rows.append(
                {
                    "Season": season,
                    "Component": component,
                    "TeamGameMean": mean,
                    "TeamGameSD": sd,
                    "StandardizedMeanShiftFromPreviousSeason": standardized_shift,
                }
            )
            previous[component] = (mean, sd)
    return pd.DataFrame(audit_rows), pd.DataFrame(distribution_rows)


def sensitivity_analysis(
    pairs: pd.DataFrame,
    primary_models: pd.DataFrame,
    player_sum_pairs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    tuning_frames: list[pd.DataFrame] = []
    target = pairs[pairs["Season"] == "E2025"].copy()
    primary = primary_models[primary_models["TargetSeason"] == "E2025"].iloc[0]
    primary_seasons = ("E2022", "E2023", "E2024")

    specifications: list[dict[str, object]] = [
        {"Analysis": "Primary", "Seasons": primary_seasons, "UsePrimarySelection": True},
        {"Analysis": "Raw bounds [0.25,1.75]", "Seasons": primary_seasons, "Lower": 0.25, "Upper": 1.75},
        {"Analysis": "Raw bounds [0.5,2.0]", "Seasons": primary_seasons, "Lower": 0.5, "Upper": 2.0},
        {"Analysis": "One-season window", "Seasons": ("E2024",)},
        {"Analysis": "Two-season window", "Seasons": ("E2023", "E2024")},
        {"Analysis": "Expanded tau-alpha grid", "Seasons": primary_seasons, "TauGrid": EXPANDED_TAU_GRID, "AlphaGrid": EXPANDED_ALPHA_GRID},
        {"Analysis": "Multi-cutoff rolling-origin selection", "Seasons": primary_seasons, "MultiCutoff": True},
        {"Analysis": "Overtime excluded", "Seasons": primary_seasons, "NoOT": True},
        {"Analysis": "Pace-normalized calibration contrasts", "Seasons": primary_seasons, "Pace": True},
        {"Analysis": "Unrestricted action groups", "Seasons": primary_seasons, "Expansion": np.eye(10)},
        {"Analysis": "Centered-SD scale anchor", "Seasons": primary_seasons, "ScaleAnchor": "Centered SD", "UsePrimarySelection": True},
        {"Analysis": "Mean-absolute scale anchor", "Seasons": primary_seasons, "ScaleAnchor": "Mean absolute contrast", "UsePrimarySelection": True},
        {"Analysis": "Leave E2022 out", "Seasons": ("E2023", "E2024")},
        {"Analysis": "Leave E2023 out", "Seasons": ("E2022", "E2024")},
        {"Analysis": "Leave E2024 out", "Seasons": ("E2022", "E2023")},
    ]
    primary_final: np.ndarray | None = None
    primary_gamma = np.nan
    for specification in specifications:
        calibration = pairs[pairs["Season"].isin(specification["Seasons"])].copy()
        evaluation = target.copy()
        if specification.get("NoOT", False):
            calibration = calibration[calibration["GameMinutes"] <= 40].copy()
            evaluation = evaluation[evaluation["GameMinutes"] <= 40].copy()
        expansion = np.asarray(specification.get("Expansion", EXPANSION), dtype=float)
        pace = bool(specification.get("Pace", False))
        lower = float(specification.get("Lower", LOWER))
        upper = float(specification.get("Upper", UPPER))
        selection_protocol = "Primary nested 70/30"
        if specification.get("UsePrimarySelection", False):
            tau, alpha = float(primary["Tau"]), float(primary["Alpha"])
        elif specification.get("MultiCutoff", False):
            tau, alpha, grid = select_hyperparameters_multicutoff(calibration, expansion, pace)
            selection_protocol = "Three-cutoff forward chaining"
            grid.insert(0, "Analysis", specification["Analysis"])
            grid.insert(1, "SelectionProtocol", selection_protocol)
            grid["Selected"] = (grid["Tau"] == tau) & (grid["Alpha"] == alpha)
            tuning_frames.append(grid)
        else:
            tau, alpha, _, grid, _, _ = select_hyperparameters(
                calibration,
                expansion,
                pace,
                specification.get("TauGrid", TAU_GRID),
                specification.get("AlphaGrid", ALPHA_GRID),
            )
            selection_protocol = "Chronological 70/30"
            grid.insert(0, "Analysis", specification["Analysis"])
            grid.insert(1, "SelectionProtocol", selection_protocol)
            grid["Selected"] = (grid["Tau"] == tau) & (grid["Alpha"] == alpha)
            tuning_frames.append(grid)
        raw_group, audit = fit_pairs(calibration, alpha, tau, lower, upper, expansion, pace)
        raw = expansion @ raw_group
        scale_anchor = str(specification.get("ScaleAnchor", "RMS"))
        if scale_anchor == "RMS":
            gamma, _ = rms_anchor(calibration, raw, pace)
        else:
            gamma = alternative_scale_anchor(calibration, raw, scale_anchor)
        final = gamma * raw
        if specification["Analysis"] == "Primary":
            primary_final = final.copy()
            primary_gamma = gamma
        matrix = design_matrix(evaluation)
        official_np = matrix @ np.ones(10)
        proposed_np = matrix @ final
        official_full = evaluation["DeltaPTS"].to_numpy(float) + official_np
        proposed_full = evaluation["DeltaPTS"].to_numpy(float) + proposed_np
        for estimand, official, proposed in (
            ("Non-PTS", official_np, proposed_np),
            ("Full", official_full, proposed_full),
        ):
            oc = classification(official)
            pc = classification(proposed)
            rows.append(
                {
                    "Analysis": specification["Analysis"],
                    "AnalysisRole": "Primary" if specification["Analysis"] == "Primary" else "Retrospective sensitivity",
                    "SelectionProtocol": selection_protocol,
                    "ScaleAnchor": scale_anchor,
                    "CalibrationSeasons": "+".join(specification["Seasons"]),
                    "Estimand": estimand,
                    "CalibrationGames": len(calibration),
                    "TargetGames": len(evaluation),
                    "Tau": tau,
                    "Alpha": alpha,
                    "LowerBound": lower,
                    "UpperBound": upper,
                    "Gamma": gamma,
                    "OfficialWinnerHigher": oc[0],
                    "VPWinnerHigher": pc[0],
                    "OfficialErrorPercent": 100.0 * (oc[1] + oc[2]) / len(evaluation),
                    "VPErrorPercent": 100.0 * (pc[1] + pc[2]) / len(evaluation),
                    "ErrorDifferencePP": 100.0 * ((pc[1] + pc[2]) - (oc[1] + oc[2])) / len(evaluation),
                    "BoundaryCount": audit["BoundaryCount"],
                    "Converged": audit["Converged"],
                }
            )

    if primary_final is None:
        raise RuntimeError("Primary sensitivity coefficients were not retained")

    # Evaluation-stage sensitivities use the primary locked E2025 coefficients.
    calibration = pairs[pairs["Season"].isin(primary_seasons)]
    evaluation_sets = (
        ("Regular-season target only", target[target["Phase"] == "Regular Season"]),
        ("Team-only residuals excluded", player_sum_pairs[player_sum_pairs["Season"] == "E2025"]),
    )
    for analysis, evaluation in evaluation_sets:
        matrix = design_matrix(evaluation)
        for estimand, official, proposed in (
            ("Non-PTS", matrix @ np.ones(10), matrix @ primary_final),
            (
                "Full",
                evaluation["DeltaPTS"].to_numpy(float) + matrix @ np.ones(10),
                evaluation["DeltaPTS"].to_numpy(float) + matrix @ primary_final,
            ),
        ):
            oc, pc = classification(official), classification(proposed)
            rows.append(
                {
                    "Analysis": analysis,
                    "AnalysisRole": "Retrospective sensitivity",
                    "SelectionProtocol": "Primary nested 70/30",
                    "ScaleAnchor": "RMS",
                    "CalibrationSeasons": "+".join(primary_seasons),
                    "Estimand": estimand,
                    "CalibrationGames": len(calibration),
                    "TargetGames": len(evaluation),
                    "Tau": primary["Tau"],
                    "Alpha": primary["Alpha"],
                    "LowerBound": LOWER,
                    "UpperBound": UPPER,
                    "Gamma": primary_gamma,
                    "OfficialWinnerHigher": oc[0],
                    "VPWinnerHigher": pc[0],
                    "OfficialErrorPercent": 100.0 * (oc[1] + oc[2]) / len(evaluation),
                    "VPErrorPercent": 100.0 * (pc[1] + pc[2]) / len(evaluation),
                    "ErrorDifferencePP": 100.0 * ((pc[1] + pc[2]) - (oc[1] + oc[2])) / len(evaluation),
                    "BoundaryCount": primary["BoundaryCount"],
                    "Converged": True,
                }
            )
    tuning = pd.concat(tuning_frames, ignore_index=True, sort=False)
    return pd.DataFrame(rows), tuning


def optimization_audit(pairs: pd.DataFrame, models: pd.DataFrame, tuning: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    first = models.iloc[0]
    calibration = pairs[pairs["Season"].isin(str(first["CalibrationSeasons"]).split("+"))]
    grouped = design_matrix(calibration) @ EXPANSION
    alpha, tau = float(first["Alpha"]), float(first["Tau"])

    def objective(a: np.ndarray) -> float:
        z = grouped @ a
        return float(np.mean(np.logaddexp(0.0, -z / tau)) + alpha * np.mean(np.square(EXPANSION @ a - 1.0)))

    def gradient(a: np.ndarray) -> np.ndarray:
        z = grouped @ a
        return -(grouped.T @ expit(-z / tau)) / (len(grouped) * tau) + (
            2.0 * alpha * EXPANSION.T @ (EXPANSION @ a - 1.0) / len(COMPONENTS)
        )

    gradient_error = float(check_grad(objective, gradient, np.ones(len(GROUPS))))
    rows.append({"Check": "Analytic versus finite-difference gradient", "Value": gradient_error, "Passed": gradient_error < 1e-5})
    rows.append({"Check": "All tuning fits converged", "Value": int(tuning["Converged"].sum()), "Passed": bool(tuning["Converged"].all())})
    rows.append({"Check": "All RMS identities within tolerance", "Value": float(models["RMSIdentityError"].max()), "Passed": bool((models["RMSIdentityError"] < 1e-10).all())})
    rows.append({"Check": "No optimizer fallback in final fits", "Value": int(models["Fallback"].sum()), "Passed": bool((~models["Fallback"]).all())})
    for model in models.itertuples(index=False):
        calibration = pairs[pairs["Season"].isin(str(model.CalibrationSeasons).split("+"))]
        solutions = []
        for start in (np.full(len(GROUPS), LOWER), np.ones(len(GROUPS)), np.full(len(GROUPS), UPPER)):
            raw, _ = fit_pairs(calibration, float(model.Alpha), float(model.Tau), start=start)
            solutions.append(raw)
        maximum_difference = float(np.ptp(np.vstack(solutions), axis=0).max())
        rows.append(
            {
                "Check": f"Multiple-start coefficient agreement {model.TargetSeason}",
                "Value": maximum_difference,
                "Passed": maximum_difference < 1e-5,
            }
        )
    return pd.DataFrame(rows)


def tex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def signed_integer(value: int) -> str:
    return f"+{value}" if value > 0 else str(value)


def player_display_name(value: object) -> str:
    """Convert the official all-capitals display field without corrupting suffixes or Mc names."""

    text = re.sub(r"\s+", " ", str(value).strip())
    text = re.sub(r"\s+,", ",", text).title()
    text = re.sub(r"\bMc([a-z])", lambda match: "Mc" + match.group(1).upper(), text)
    text = re.sub(r"\b(Ii|Iii|Iv|Vi)\b", lambda match: match.group(1).upper(), text)
    text = re.sub(r"(?<=, )([A-Za-z]{2})$", lambda match: match.group(1).upper(), text)
    text = text.replace("Dibartolomeo", "DiBartolomeo")
    return text


def tex_scientific(value: float, digits: int = 2) -> str:
    if value == 0:
        return "0"
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / (10.0**exponent)
    return rf"{mantissa:.{digits}f}\times10^{{{exponent}}}"


def tex_p_value(value: float) -> str:
    return rf"${tex_scientific(value)}$" if value < 0.001 else f"{value:.3f}"


def write_generated_tables(
    players: pd.DataFrame,
    teams: pd.DataFrame,
    teams_all_stages: pd.DataFrame,
    validation: pd.DataFrame,
) -> None:
    player_lines = [
        r"\begingroup",
        r"\setlength{\tabcolsep}{2.6pt}",
        r"\scriptsize",
        r"\begin{longtable}{@{}p{0.200\textwidth}p{0.090\textwidth}rrrrrrr@{}}",
        rf"\caption{{All {len(players)} players with a positive-minute appearance in 2025--26. Official PIR and RM-PIR are all-stage season totals; the RM-PIR coefficients were estimated from 2022--23 through 2024--25 and locked before the target season. GP denotes games played and MIN total minutes. Ranks use competition ranking on season totals. Rank change is official rank minus RM-PIR rank, so positive values indicate upward movement. Team(s) are official club codes and list every club represented by a transferred player during 2025--26.}}\label{{tab:players-e2025}}\\",
        r"\toprule",
        r"Player & Team(s) & GP & MIN & PIR & RM-PIR & PIR rank & VP rank & $\Delta$ rank \\",
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{9}{l}{\textit{Table 2 continued}}\\",
        r"\toprule",
        r"Player & Team(s) & GP & MIN & PIR & RM-PIR & PIR rank & VP rank & $\Delta$ rank \\",
        r"\midrule",
        r"\endhead",
        r"\midrule",
        r"\multicolumn{9}{r}{\textit{Continued on next page}}\\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    for row in players.itertuples(index=False):
        player_lines.append(
            "{} & {} & {} & {:.0f} & {:.0f} & {:.2f} & {} & {} & {} \\\\".format(
                tex_escape(player_display_name(row.Player)),
                tex_escape(row.Teams),
                row.GP,
                row.Minutes,
                row.OfficialPIR,
                row.VPPIR,
                row.OfficialRank,
                row.VPRank,
                signed_integer(int(row.RankChange)),
            )
        )
    player_lines.extend([r"\end{longtable}", r"\endgroup"])
    (TABLES / "table2_players_e2025.tex").write_text("\n".join(player_lines) + "\n", encoding="utf-8")

    team_lines = [
        r"\begin{table*}[t!]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        rf"\caption{{2025--26 regular-season team rankings. GP denotes games played. Every club played {int(teams['GP'].iloc[0])} games, so ranking by total or per-game mean is identical. Rank change is official PIR rank minus RM-PIR rank; positive values indicate upward movement. RM-PIR used coefficients selected and fitted exclusively from 2022--23 through 2024--25.}}",
        r"\label{tab:teams-e2025}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{@{}llrrrrrrrr@{}}",
        r"\toprule",
        r"Team & Code & GP & PIR total & VP total & PIR/game & VP/game & PIR rank & VP rank & $\Delta$ rank \\",
        r"\midrule",
    ]
    for row in teams.itertuples(index=False):
        team_lines.append(
            "{} & {} & {} & {:.0f} & {:.2f} & {:.2f} & {:.2f} & {} & {} & {} \\\\".format(
                tex_escape(row.Team),
                tex_escape(row.TeamCode),
                row.GP,
                row.OfficialPIRTotal,
                row.VPPIRTotal,
                row.OfficialPIRPerGame,
                row.VPPIRPerGame,
                row.OfficialRank,
                row.VPRank,
                signed_integer(int(row.RankChange)),
            )
        )
    team_lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}"])
    (TABLES / "table3_teams_e2025.tex").write_text("\n".join(team_lines) + "\n", encoding="utf-8")

    all_stage_team_lines = [
        r"\begin{table*}[t!]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{2025--26 all-stage team rankings over all 402 completed games (380 regular-season, three play-in, 16 playoff and three Final Four games). GP denotes games played. In contrast to Table~\ref{tab:teams-e2025}, which restricts every club to the common 38-game regular season, this table includes all completed competition stages; games played consequently range from 38 to 44 because postseason participation is unequal. Ranks are based on cumulative totals and therefore reflect both per-game performance and postseason exposure. The adjacent per-game columns separate scoring rate from cumulative opportunity. Rank change is official PIR rank minus RM-PIR rank, so positive values indicate upward movement. RM-PIR used coefficients selected and fitted exclusively from 2022--23 through 2024--25.}",
        r"\label{tab:teams-e2025-all-stages}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{@{}llrrrrrrrr@{}}",
        r"\toprule",
        r"Team & Code & GP & PIR total & VP total & PIR/game & VP/game & PIR rank & VP rank & $\Delta$ rank \\",
        r"\midrule",
    ]
    all_stage_display = teams_all_stages.sort_values(
        ["VPTotalRank", "OfficialTotalRank", "Team"], ascending=[True, True, True]
    )
    for row in all_stage_display.itertuples(index=False):
        all_stage_team_lines.append(
            "{} & {} & {} & {:.0f} & {:.2f} & {:.2f} & {:.2f} & {} & {} & {} \\\\".format(
                tex_escape(row.Team),
                tex_escape(row.TeamCode),
                row.GP,
                row.OfficialPIRTotal,
                row.VPPIRTotal,
                row.OfficialPIRPerGame,
                row.VPPIRPerGame,
                row.OfficialTotalRank,
                row.VPTotalRank,
                signed_integer(int(row.TotalRankChange)),
            )
        )
    all_stage_team_lines.extend(
        [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}"]
    )
    (TABLES / "table4_teams_e2025_all_stages.tex").write_text(
        "\n".join(all_stage_team_lines) + "\n", encoding="utf-8"
    )

    validation_lines = [
        r"\begin{table*}[t!]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.4pt}",
        r"\caption{Nested rolling-origin validation across five target seasons. W/T/L denotes winner higher, numerical tie and loser higher at tolerance $10^{-9}$; ties and reversals count as ordering errors. Error-rate brackets are Wilson 95\% intervals. Paired cells give, in order, both metrics concordant, official PIR only concordant (a new RM-PIR discordance), RM-PIR only concordant (a correction), and both metrics discordant. $P$ is the two-sided exact McNemar value. Within each eligible calibration window, $(\tau,\alpha)$ was selected on a chronological 70/30 split of distinct dates and then refitted on the full window before RMS scaling and target-season locking. $\Delta$ error is RM-PIR minus official PIR in percentage points; its paired-game bootstrap 95\% interval is shown in brackets. SRE is unsigned and secondary. Pooled rows are descriptive; the protocol was retrospectively specified rather than preregistered.}",
        r"\label{tab:rolling-validation}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{@{}llrlllllrrrrr@{}}",
        r"\toprule",
        r"Target & Calibration & Games & PIR W/T/L & VP W/T/L & PIR err. [95\% CI] & VP err. [95\% CI] & Paired cells & $\Delta$ err. [95\% CI] & $P$ & SRE & $\tau$ & $\alpha$ \\",
        r"\midrule",
    ]
    for estimand in ("Non-PTS", "Full"):
        estimand_label = "Complete" if estimand == "Full" else estimand
        validation_lines.append(rf"\multicolumn{{13}}{{l}}{{\textit{{{estimand_label} score}}}}\\")
        subset = validation[validation["Estimand"] == estimand]
        for row in subset.itertuples(index=False):
            target = "Pooled" if row.TargetSeason == "Pooled" else SEASON_LABELS[row.TargetSeason]
            if row.TargetSeason == "Pooled":
                calibration = "--"
            else:
                calibration_codes = row.CalibrationSeasons.split("+")
                calibration = SEASON_LABELS[calibration_codes[0]]
                if len(calibration_codes) > 1:
                    calibration += " to " + SEASON_LABELS[calibration_codes[-1]]
            tau = "--" if pd.isna(row.Tau) else f"{row.Tau:g}"
            alpha = "--" if pd.isna(row.Alpha) else f"{row.Alpha:g}"
            both_failure = int(row.VPTies + row.VPLoserHigher - row.NewFailures)
            both_success = int(row.Games - row.Corrections - row.NewFailures - both_failure)
            validation_lines.append(
                "{} & {} & {} & {}/{}/{} & {}/{}/{} & {:.2f} [{:.2f}, {:.2f}] & {:.2f} [{:.2f}, {:.2f}] & {}/{}/{}/{} & {:.2f} [{:.2f}, {:.2f}] & {} & {:.1f} & {} & {} \\\\".format(
                    target,
                    tex_escape(calibration),
                    row.Games,
                    row.OfficialWinnerHigher,
                    row.OfficialTies,
                    row.OfficialLoserHigher,
                    row.VPWinnerHigher,
                    row.VPTies,
                    row.VPLoserHigher,
                    row.OfficialErrorRatePercent,
                    row.OfficialErrorRateCI2_5Percent,
                    row.OfficialErrorRateCI97_5Percent,
                    row.VPErrorRatePercent,
                    row.VPErrorRateCI2_5Percent,
                    row.VPErrorRateCI97_5Percent,
                    both_success,
                    row.NewFailures,
                    row.Corrections,
                    both_failure,
                    row.ErrorRateDifferencePP,
                    row.ErrorRateDifferenceCI2_5PP,
                    row.ErrorRateDifferenceCI97_5PP,
                    tex_p_value(row.McNemarExactP),
                    row.SymmetricRelativeErrorPercent,
                    tau,
                    alpha,
                )
            )
        if estimand == "Non-PTS":
            validation_lines.append(r"\addlinespace")
    validation_lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}"])
    (TABLES / "table5_rolling_validation.tex").write_text(
        "\n".join(validation_lines) + "\n", encoding="utf-8"
    )


def write_manuscript_values(
    validation: pd.DataFrame,
    models: pd.DataFrame,
    players: pd.DataFrame,
    teams: pd.DataFrame,
    residual_identity: pd.DataFrame,
    comparability: pd.DataFrame,
    optimizer: pd.DataFrame,
    sensitivity: pd.DataFrame,
    group_diagnostics: pd.DataFrame,
    team_dependence: pd.DataFrame,
) -> None:
    focal_np = validation[(validation["TargetSeason"] == "E2025") & (validation["Estimand"] == "Non-PTS")].iloc[0]
    focal_full = validation[(validation["TargetSeason"] == "E2025") & (validation["Estimand"] == "Full")].iloc[0]
    pooled_np = validation[(validation["TargetSeason"] == "Pooled") & (validation["Estimand"] == "Non-PTS")].iloc[0]
    pooled_full = validation[(validation["TargetSeason"] == "Pooled") & (validation["Estimand"] == "Full")].iloc[0]
    focal_model = models[models["TargetSeason"] == "E2025"].iloc[0]
    olympiacos = teams[teams["TeamCode"] == "OLY"].iloc[0]
    largest_shift = comparability["StandardizedMeanShiftFromPreviousSeason"].max()
    max_identity_error = residual_identity["VPAdditivityIdentityMaximumAbsoluteError"].dropna().max()
    gradient_error = optimizer.loc[
        optimizer["Check"] == "Analytic versus finite-difference gradient", "Value"
    ].iloc[0]
    maximum_start_difference = optimizer.loc[
        optimizer["Check"].str.startswith("Multiple-start coefficient agreement"), "Value"
    ].max()
    focal_groups = group_diagnostics[group_diagnostics["TargetSeason"] == "E2025"].set_index("Pair")
    omissions = team_dependence[team_dependence["OmittedTeam"] != "None"]
    omission_nonpoint = omissions[omissions["Estimand"] == "Non-PTS"]["ErrorDifferencePP"]
    omission_full = omissions[omissions["Estimand"] == "Full"]["ErrorDifferencePP"]
    sensitivity_analyses = sensitivity.loc[
        sensitivity["AnalysisRole"] == "Retrospective sensitivity", "Analysis"
    ].nunique()

    values = {
        "TotalSixSeasonGames": f"{sum(EXPECTED_GAMES.values()):,}",
        "TotalTargetGames": f"{sum(EXPECTED_GAMES[s] for s in EXPERIMENTS):,}",
        "TotalTeamRows": f"{2 * sum(EXPECTED_GAMES.values()):,}",
        "ESeasonGames": f"{EXPECTED_GAMES['E2025']:,}",
        "ESeasonRegularGamesPerTeam": str(int(teams["GP"].iloc[0])),
        "ESeasonPlayerCount": str(len(players)),
        "ESeasonTeamCount": str(len(teams)),
        "ESeasonTau": f"{focal_model['Tau']:g}",
        "ESeasonAlpha": f"{focal_model['Alpha']:g}",
        "ESeasonGamma": f"{focal_model['Gamma']:.4f}",
        "ESeasonFinalCoefficientMinimum": f"{focal_model['FinalMagnitudeMinimum']:.3f}",
        "ESeasonFinalCoefficientMaximum": f"{focal_model['FinalMagnitudeMaximum']:.3f}",
        "ESeasonOfficialRMS": f"{focal_model['OfficialNonpointRMS']:.3f}",
        "ESeasonFinalRMS": f"{focal_model['FinalNonpointRMS']:.3f}",
        "ESeasonNonpointOfficialErrors": str(int(focal_np.OfficialTies + focal_np.OfficialLoserHigher)),
        "ESeasonNonpointVPErrors": str(int(focal_np.VPTies + focal_np.VPLoserHigher)),
        "ESeasonNonpointErrorDifference": f"{focal_np.ErrorRateDifferencePP:.2f}",
        "ESeasonNonpointCorrections": str(int(focal_np.Corrections)),
        "ESeasonNonpointNewFailures": str(int(focal_np.NewFailures)),
        "ESeasonNonpointMcNemarP": tex_scientific(float(focal_np.McNemarExactP)),
        "ESeasonFullOfficialErrors": str(int(focal_full.OfficialTies + focal_full.OfficialLoserHigher)),
        "ESeasonFullVPErrors": str(int(focal_full.VPTies + focal_full.VPLoserHigher)),
        "ESeasonFullErrorDifference": f"{focal_full.ErrorRateDifferencePP:.2f}",
        "ESeasonFullCorrections": str(int(focal_full.Corrections)),
        "ESeasonFullNewFailures": str(int(focal_full.NewFailures)),
        "ESeasonFullMcNemarP": f"{focal_full.McNemarExactP:.3f}",
        "PooledNonpointOfficialErrors": str(int(pooled_np.OfficialTies + pooled_np.OfficialLoserHigher)),
        "PooledNonpointVPErrors": str(int(pooled_np.VPTies + pooled_np.VPLoserHigher)),
        "PooledNonpointErrorDifference": f"{pooled_np.ErrorRateDifferencePP:.2f}",
        "PooledFullOfficialErrors": str(int(pooled_full.OfficialTies + pooled_full.OfficialLoserHigher)),
        "PooledFullVPErrors": str(int(pooled_full.VPTies + pooled_full.VPLoserHigher)),
        "PooledFullErrorDifference": f"{pooled_full.ErrorRateDifferencePP:.2f}",
        "OlympiacosOfficialRank": str(int(olympiacos.OfficialRank)),
        "OlympiacosVPRank": str(int(olympiacos.VPRank)),
        "OlympiacosRankChange": signed_integer(int(olympiacos.RankChange)),
        "BlockPairCorrelation": f"{focal_groups.loc['BLK-BLKA', 'SignedContrastPearsonR']:.5f}",
        "FoulPairCorrelation": f"{focal_groups.loc['FD-FC', 'SignedContrastPearsonR']:.5f}",
        "SensitivityAnalysisCount": str(int(sensitivity_analyses)),
        "NonpointTeamOmissionMinimum": f"{omission_nonpoint.min():.2f}",
        "NonpointTeamOmissionMaximum": f"{omission_nonpoint.max():.2f}",
        "FullTeamOmissionMinimum": f"{omission_full.min():.2f}",
        "FullTeamOmissionMaximum": f"{omission_full.max():.2f}",
        "MaximumResidualIdentityError": tex_scientific(max_identity_error),
        "LargestAdjacentSeasonStandardizedShift": f"{largest_shift:.2f}",
        "TuningFitCount": f"{len(EXPERIMENTS) * len(TAU_GRID) * len(ALPHA_GRID):,}",
        "MaximumRMSIdentityError": tex_scientific(float(models["RMSIdentityError"].max())),
        "GradientCheckError": tex_scientific(gradient_error),
        "MaximumMultipleStartDifference": tex_scientific(maximum_start_difference),
    }
    lines = ["% Generated by professor_revision_analysis.py; do not edit by hand."]
    lines.extend(rf"\newcommand{{\{name}}}{{{value}}}" for name, value in values.items())
    (DERIVED / "manuscript_values.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def figure_methodology() -> None:
    fig, ax = plt.subplots(figsize=(10.2, 3.2))
    ax.set_xlim(0, 10.2)
    ax.set_ylim(0, 3.2)
    ax.axis("off")
    boxes = [
        (0.15, 1.15, 1.55, 1.15, "Completed prior\nteam box scores"),
        (2.05, 1.15, 1.65, 1.15, "Winner--loser\nnon-PTS contrasts"),
        (4.05, 1.15, 1.75, 1.15, "Nested tuning and\nfull-window refit"),
        (6.15, 1.15, 1.55, 1.15, "Calibration-window\nRMS scale anchor"),
        (8.05, 1.15, 1.85, 1.15, "Season-locked RM-PIR\nfor target teams/players"),
    ]
    for index, (x, y, w, h, label) in enumerate(boxes):
        face = "#eaf4fb" if index < 4 else "#d5e9f6"
        patch = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.07",
            facecolor=face, edgecolor="#2171b5", linewidth=1.2
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=9)
    for index in range(len(boxes) - 1):
        x1 = boxes[index][0] + boxes[index][2]
        x2 = boxes[index + 1][0]
        ax.add_patch(FancyArrowPatch((x1 + 0.04, 1.72), (x2 - 0.04, 1.72), arrowstyle="-|>", mutation_scale=11, color="#4d4d4d", linewidth=1.0))
    ax.text(4.92, 2.75, r"Primary calibration estimand: $z_g(\widehat{\mathbf{a}})>0$", ha="center", fontsize=10, weight="bold")
    ax.text(8.97, 0.66, r"$\Delta_g^{\mathrm{VP}}=\Delta_g^{\mathrm{PTS}}+\gamma z_g(\widehat{\mathbf{a}})$", ha="center", fontsize=10)
    ax.annotate("PTS coefficient fixed at 1; PTS absent from fitting", xy=(8.15, 1.15), xytext=(6.1, 0.25), arrowprops={"arrowstyle": "->", "color": "#d95f0e"}, color="#a63603", fontsize=8.5)
    fig.savefig(FIGURES / "vppir_methodology.pdf", metadata={"Title": "Nested season-locked RM-PIR calibration protocol"})
    plt.close(fig)


def figure_validation(validation: pd.DataFrame) -> None:
    focal = validation[validation["TargetSeason"] == "E2025"].set_index("Estimand")
    labels = ["Non-PTS\n(primary)", "Complete score\n(downstream)"]
    official = [focal.loc["Non-PTS", "OfficialErrorRatePercent"], focal.loc["Full", "OfficialErrorRatePercent"]]
    proposed = [focal.loc["Non-PTS", "VPErrorRatePercent"], focal.loc["Full", "VPErrorRatePercent"]]
    x = np.arange(2)
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    ax.bar(x - 0.19, official, 0.38, label="Official PIR", color="#9ecae1")
    ax.bar(x + 0.19, proposed, 0.38, label="RM-PIR", color="#2171b5")
    for positions, values in ((x - 0.19, official), (x + 0.19, proposed)):
        for position, value in zip(positions, values):
            ax.text(position, value + 0.25, f"{value:.2f}%", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Ordering error (% of 2025–26 games)")
    ax.set_ylim(0, max(official + proposed) * 1.25)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    ax.grid(axis="y", color="#ececec", linewidth=0.6)
    fig.savefig(FIGURES / "vppir_validation.pdf", metadata={"Title": "2025–26 RM-PIR ordering error"})
    plt.close(fig)


def figure_changed_games(games: pd.DataFrame) -> None:
    focal = games[games["Season"] == "E2025"].copy()
    corrected = (focal["OfficialFullDiff"] <= TOLERANCE) & (focal["VPFullDiff"] > TOLERANCE)
    new_failure = (focal["OfficialFullDiff"] > TOLERANCE) & (focal["VPFullDiff"] <= TOLERANCE)
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    ax.scatter(focal["OfficialFullDiff"], focal["VPFullDiff"], s=18, color="#9ECAE1", alpha=0.70, edgecolors="none", zorder=1, label="All games")
    ax.scatter(focal.loc[corrected, "OfficialFullDiff"], focal.loc[corrected, "VPFullDiff"], s=34, color="#2171B5", edgecolors="white", linewidths=0.45, zorder=2, label="Official discordance corrected")
    if new_failure.any():
        ax.scatter(focal.loc[new_failure, "OfficialFullDiff"], focal.loc[new_failure, "VPFullDiff"], s=38, marker="D", color="#D95F0E", edgecolors="white", linewidths=0.45, zorder=3, label="New RM-PIR discordance")
    limits = [min(focal["OfficialFullDiff"].min(), focal["VPFullDiff"].min()), max(focal["OfficialFullDiff"].max(), focal["VPFullDiff"].max())]
    padding = 0.04 * (limits[1] - limits[0])
    limits = [limits[0] - padding, limits[1] + padding]
    ax.plot(limits, limits, linestyle="--", color="#666666", linewidth=0.8, zorder=0)
    ax.axhline(0, color="black", linewidth=0.85)
    ax.axvline(0, color="black", linewidth=0.85)
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_xlabel("Official PIR winner-minus-loser difference")
    ax.set_ylabel("RM-PIR winner-minus-loser difference")
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.savefig(FIGURES / "vppir_changed_games.pdf", metadata={"Title": "E2025 official PIR and RM-PIR game differences"})
    plt.close(fig)


def figure_team_ranks(teams: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 7.2))
    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(len(teams) + 0.7, 0.3)
    ax.axis("off")
    for row in teams.itertuples(index=False):
        if row.RankChange > 0:
            color, linewidth, alpha = "#2171b5", 1.8, 0.95
        elif row.RankChange < 0:
            color, linewidth, alpha = "#d95f0e", 1.2, 0.75
        else:
            color, linewidth, alpha = "#9e9e9e", 0.9, 0.65
        ax.plot([0, 1], [row.OfficialRank, row.VPRank], color=color, linewidth=linewidth, alpha=alpha)
        ax.scatter([0, 1], [row.OfficialRank, row.VPRank], color=color, s=18, zorder=2)
        ax.text(-0.03, row.OfficialRank, f"{row.OfficialRank:>2}  {row.TeamCode}", ha="right", va="center", fontsize=8)
        suffix = f" ({signed_integer(int(row.RankChange))})" if row.RankChange else ""
        ax.text(1.03, row.VPRank, f"{row.TeamCode}  {row.VPRank:>2}{suffix}", ha="left", va="center", fontsize=8)
    ax.text(0, 0.3, "Official PIR rank", ha="center", va="bottom", fontsize=10, weight="bold")
    ax.text(1, 0.3, "RM-PIR rank", ha="center", va="bottom", fontsize=10, weight="bold")
    ax.text(0.5, len(teams) + 0.55, "Positive parenthetical values denote upward movement", ha="center", fontsize=7.5, color="#555555")
    fig.savefig(FIGURES / "vppir_team_rank_movement.pdf", metadata={"Title": "E2025 regular-season team rank movement"})
    plt.close(fig)


def figure_coefficients(
    coefficients: pd.DataFrame, coefficient_intervals: pd.DataFrame
) -> None:
    targets = list(EXPERIMENTS)
    matrix = np.empty((len(COMPONENTS), len(targets)))
    interval_lower = np.empty_like(matrix)
    interval_upper = np.empty_like(matrix)
    boundary = np.zeros_like(matrix, dtype=bool)
    for column, target in enumerate(targets):
        frame = coefficients[coefficients["TargetSeason"] == target].set_index("Component")
        interval_frame = coefficient_intervals[
            coefficient_intervals["TargetSeason"] == target
        ].set_index("Component")
        matrix[:, column] = frame.loc[list(COMPONENTS), "FinalMagnitude"]
        lower_signed = interval_frame.loc[list(COMPONENTS), "SignedCI2_5"].to_numpy(float)
        upper_signed = interval_frame.loc[list(COMPONENTS), "SignedCI97_5"].to_numpy(float)
        interval_lower[:, column] = np.minimum(
            np.abs(lower_signed), np.abs(upper_signed)
        )
        interval_upper[:, column] = np.maximum(
            np.abs(lower_signed), np.abs(upper_signed)
        )
        boundary[:, column] = (
            frame.loc[list(COMPONENTS), "RawAtLowerBound"].to_numpy(bool)
            | frame.loc[list(COMPONENTS), "RawAtUpperBound"].to_numpy(bool)
        )
    maximum = float(interval_upper.max())
    x = np.arange(len(COMPONENTS), dtype=float)
    width = 0.145
    colors = ("#0072B2", "#56B4E9", "#009E73", "#E69F00", "#CC79A7")

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.axvspan(-0.55, 4.5, color="#eaf4fb", alpha=0.65, zorder=0)
    ax.axvspan(4.5, 9.55, color="#fff2e2", alpha=0.65, zorder=0)
    for column, (target, color) in enumerate(zip(targets, colors)):
        positions = x + (column - (len(targets) - 1) / 2.0) * width
        ax.bar(
            positions,
            matrix[:, column],
            width=0.92 * width,
            color=color,
            edgecolor="white",
            linewidth=0.35,
            yerr=np.vstack(
                (
                    matrix[:, column] - interval_lower[:, column],
                    interval_upper[:, column] - matrix[:, column],
                )
            ),
            error_kw={
                "ecolor": "#333333",
                "elinewidth": 0.55,
                "capsize": 1.2,
                "capthick": 0.55,
            },
            label=SEASON_LABELS[target],
            zorder=2,
        )
        active = boundary[:, column]
        ax.scatter(
            positions[active],
            matrix[active, column] + 0.045,
            s=18,
            facecolors="none",
            edgecolors="black",
            linewidths=0.75,
            zorder=5,
        )

    ax.axhline(
        1.0,
        color="#222222",
        linewidth=1.15,
        linestyle=(0, (5, 3)),
        zorder=4,
    )
    ax.axvline(4.5, color="#555555", linewidth=1.0, zorder=4)
    ax.text(
        9.48,
        1.035,
        "Official PIR weight",
        ha="right",
        va="bottom",
        fontsize=8,
        color="#222222",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.5},
        zorder=6,
    )
    ax.text(
        2.0,
        1.025,
        "Positive action credits",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=9,
        weight="bold",
        color="#1f5f89",
    )
    ax.text(
        7.0,
        1.025,
        "Penalty components",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=9,
        weight="bold",
        color="#9a5b13",
    )
    ax.set_xticks(x, COMPONENTS)
    ax.set_xlim(-0.55, len(COMPONENTS) - 0.45)
    ax.set_ylim(0, maximum * 1.17)
    ax.set_ylabel(r"Final coefficient magnitude, $|c_{j,Y}|$")
    ax.set_xlabel("PIR component")
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.55, zorder=1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        title="Target season",
        frameon=False,
        ncol=len(targets),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.19),
        columnspacing=1.3,
        handlelength=1.3,
        fontsize=8,
        title_fontsize=8,
    )
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.14, top=0.77)
    fig.savefig(
        FIGURES / "vppir_coefficient_stability.pdf",
        metadata={"Title": "Season-locked RM-PIR coefficient magnitudes"},
    )
    plt.close(fig)


def figure_historical(validation: pd.DataFrame) -> None:
    seasons = list(EXPERIMENTS)
    labels = [SEASON_LABELS[season] for season in seasons]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8), gridspec_kw={"width_ratios": [1.15, 1.0]})
    colors = {"Non-PTS": "#2171b5", "Full": "#d95f0e"}
    markers = {"Official": "o", "VP": "s"}
    for estimand in ("Non-PTS", "Full"):
        frame = validation[(validation["Estimand"] == estimand) & (validation["TargetSeason"] != "Pooled")].set_index("TargetSeason").loc[seasons]
        axes[0].plot(labels, frame["OfficialErrorRatePercent"], marker=markers["Official"], linestyle="--", color=colors[estimand], alpha=0.55, label=f"Official {estimand}")
        axes[0].plot(labels, frame["VPErrorRatePercent"], marker=markers["VP"], linestyle="-", color=colors[estimand], label=f"RM-PIR {estimand}")
        offset = -0.17 if estimand == "Non-PTS" else 0.17
        x = np.arange(len(seasons)) + offset
        axes[1].bar(x, frame["ErrorRateDifferencePP"], width=0.34, color=colors[estimand], label=estimand)
    axes[0].set_title("a  Target-season ordering error")
    axes[0].set_ylabel("Error (% of games)")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].grid(axis="y", color="#ececec", linewidth=0.6)
    axes[0].legend(frameon=False, fontsize=7, ncol=2)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xticks(np.arange(len(seasons)), labels, rotation=30)
    axes[1].set_title("b  Signed error-rate change")
    axes[1].set_ylabel("RM-PIR minus official PIR (percentage points)")
    axes[1].legend(frameon=False, fontsize=7)
    axes[1].grid(axis="y", color="#ececec", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(FIGURES / "vppir_historical_validation.pdf", metadata={"Title": "Five-target-season nested rolling RM-PIR validation"})
    plt.close(fig)


def write_manifest(inputs: list[Path], outputs: list[Path]) -> None:
    manifest_path = DERIVED / "analysis_manifest.json"
    previous = {}
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = {
        "analysis": "Season-locked RMS-matched PIR rolling backtest and technical-revision audits",
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "seed": SEED,
        "tie_tolerance": TOLERANCE,
        "rms_denominator_safeguard": RMS_EPSILON,
        "paired_bootstraps": PAIRED_BOOTSTRAPS,
        "coefficient_bootstraps_per_target": COEFFICIENT_BOOTSTRAPS,
        "inputs": {str(path.relative_to(ROOT)): sha256(path) for path in inputs},
        "outputs": {str(path.relative_to(ROOT)): sha256(path) for path in outputs if path.exists()},
    }
    if "raw_stats_collections" in previous:
        manifest["raw_stats_collections"] = previous["raw_stats_collections"]
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    DERIVED.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata()
    names = load_team_names()
    totals = load_team_totals(metadata)
    players = load_players()
    pairs = build_pairs(totals)

    validation, transitions, models, coefficients, games, tuning = rolling_analysis(totals, pairs)
    coefficient_intervals, bootstrap_performance = coefficient_bootstrap(pairs, models)
    player_table, team_regular, team_all_stage = player_and_team_outputs(totals, players, coefficients, names)
    residual_components, residual_identity = residual_audit(totals, players, coefficients)
    player_sum_pairs = build_player_sum_pairs(totals, players)
    data_audit, comparability = data_and_comparability_audit(totals, pairs)
    sensitivity, sensitivity_tuning = sensitivity_analysis(pairs, models, player_sum_pairs)
    group_diagnostics = group_restriction_diagnostics(pairs)
    team_dependence = team_dependence_sensitivity(games)
    optimizer = optimization_audit(pairs, models, tuning)

    outputs: dict[str, pd.DataFrame] = {
        "rolling_validation_summary.csv": validation,
        "paired_transitions.csv": transitions,
        "model_registry.csv": models,
        "rolling_coefficients.csv": coefficients,
        "rolling_game_results.csv": games,
        "nested_tuning_grid.csv": tuning,
        "coefficient_bootstrap_intervals.csv": coefficient_intervals,
        "coefficient_bootstrap_target_performance.csv": bootstrap_performance,
        "e2025_players.csv": player_table,
        "e2025_teams_regular_season.csv": team_regular,
        "e2025_teams_all_stages.csv": team_all_stage,
        "residual_component_audit.csv": residual_components,
        "residual_identity_audit.csv": residual_identity,
        "data_integrity_audit.csv": data_audit,
        "event_comparability_diagnostics.csv": comparability,
        "sensitivity_summary.csv": sensitivity,
        "sensitivity_tuning_grid.csv": sensitivity_tuning,
        "group_restriction_diagnostics.csv": group_diagnostics,
        "team_dependence_sensitivity.csv": team_dependence,
        "optimization_audit.csv": optimizer,
    }
    for filename, frame in outputs.items():
        frame.to_csv(DERIVED / filename, index=False)

    figure_methodology()
    figure_validation(validation)
    figure_changed_games(games)
    figure_team_ranks(team_regular)
    figure_coefficients(coefficients, coefficient_intervals)
    figure_historical(validation)

    generated_paths = [DERIVED / filename for filename in outputs]
    generated_paths.extend(
        [
            FIGURES / "vppir_methodology.pdf",
            FIGURES / "vppir_validation.pdf",
            FIGURES / "vppir_changed_games.pdf",
            FIGURES / "vppir_team_rank_movement.pdf",
            FIGURES / "vppir_coefficient_stability.pdf",
            FIGURES / "vppir_historical_validation.pdf",
        ]
    )
    write_manifest(
        [TEAM_BOXSCORES, PLAYER_GAMES, TEAM_NAMES, SOURCE_EXCLUSIONS],
        generated_paths,
    )

    focal = validation[(validation["TargetSeason"] == "E2025") & (validation["Estimand"] == "Non-PTS")].iloc[0]
    olympiacos = team_regular[team_regular["TeamCode"] == "OLY"].iloc[0]
    print(f"Generated {len(generated_paths)} revision artifacts")
    print(
        f"E2025 non-PTS errors: official={focal.OfficialTies + focal.OfficialLoserHigher:.0f}, "
        f"RM-PIR={focal.VPTies + focal.VPLoserHigher:.0f}"
    )
    print(
        f"Olympiacos regular-season rank: official={olympiacos.OfficialRank}, "
        f"RM-PIR={olympiacos.VPRank}, change={olympiacos.RankChange:+d}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
