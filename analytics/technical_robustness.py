"""Technical checks added after the methodological review.

This script does not refit or replace the authoritative rolling-origin results.
It reads those locked results, performs the additional tie, scale, dependence,
tuning, fixed-origin, provenance, and player-association audits requested in
the paper, and regenerates the six manuscript figures that refer to the
recalibrated score.

Run from the repository root with::

    python -m analytics.technical_robustness
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from . import rm_pir_analysis as engine


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DERIVED = ROOT / "tables"
FIGURES = ROOT / "figures"
SEED = 20260810
TOLERANCE = 1e-9
EQUIVALENCE_BAND = 0.5
CLUB_BOOTSTRAPS = 20_000
JOINT_BOOTSTRAPS = 1_000
PLAYER_BOOTSTRAPS = 1_000

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
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_counts(values: np.ndarray, band: float = TOLERANCE) -> tuple[int, int, int]:
    values = np.asarray(values, dtype=float)
    counts = (
        int((values > band).sum()),
        int((np.abs(values) <= band).sum()),
        int((values < -band).sum()),
    )
    assert sum(counts) == len(values)
    return counts


def metric_columns(estimand: str) -> tuple[str, str]:
    if estimand == "Full":
        return "OfficialFullDiff", "VPFullDiff"
    if estimand == "Non-PTS":
        return "OfficialNonpointDiff", "VPNonpointDiff"
    raise ValueError(estimand)


def tie_sensitivity(games: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for target in (*engine.EXPERIMENTS.keys(), "Pooled"):
        frame = games if target == "Pooled" else games[games["Season"] == target]
        for estimand in ("Full", "Non-PTS"):
            official_column, rm_column = metric_columns(estimand)
            for score, column in (("Official PIR", official_column), ("RM-PIR", rm_column)):
                values = frame[column].to_numpy(float)
                strict = strict_counts(values)
                common = strict_counts(values, EQUIVALENCE_BAND)
                rows.append(
                    {
                        "TargetSeason": target,
                        "Estimand": estimand,
                        "Score": score,
                        "Games": len(values),
                        "WinnerHigher": strict[0],
                        "Ties": strict[1],
                        "LoserHigher": strict[2],
                        "StrictConcordancePercent": 100 * strict[0] / len(values),
                        "HalfCreditConcordancePercent": 100 * (strict[0] + 0.5 * strict[1]) / len(values),
                        "ReversalRateAllGamesPercent": 100 * strict[2] / len(values),
                        "WinnerShareAmongNonTiesPercent": 100 * strict[0] / (strict[0] + strict[2]),
                        "EquivalenceBand": EQUIVALENCE_BAND,
                        "BandWinnerHigher": common[0],
                        "BandTies": common[1],
                        "BandLoserHigher": common[2],
                        "BandStrictConcordancePercent": 100 * common[0] / len(values),
                    }
                )
    result = pd.DataFrame(rows)
    assert (result[["WinnerHigher", "Ties", "LoserHigher"]].sum(axis=1) == result["Games"]).all()
    assert (result[["BandWinnerHigher", "BandTies", "BandLoserHigher"]].sum(axis=1) == result["Games"]).all()
    return result


def figure_category_audit(games: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for target in engine.EXPERIMENTS:
        frame = games[games["Season"] == target]
        for estimand in ("Full", "Non-PTS"):
            official_column, rm_column = metric_columns(estimand)
            official = frame[official_column].to_numpy(float) > TOLERANCE
            rm = frame[rm_column].to_numpy(float) > TOLERANCE
            cells = {
                "BothConcordant": int((official & rm).sum()),
                "OfficialOnlyConcordant": int((official & ~rm).sum()),
                "RMOnlyConcordant": int((~official & rm).sum()),
                "BothDiscordant": int((~official & ~rm).sum()),
            }
            assert sum(cells.values()) == len(frame)
            assert int(rm.sum()) == int(official.sum()) + cells["RMOnlyConcordant"] - cells["OfficialOnlyConcordant"]
            rows.append({"TargetSeason": target, "Estimand": estimand, "Games": len(frame), **cells})
    return pd.DataFrame(rows)


def fixed_origin_transport(pairs: pd.DataFrame) -> pd.DataFrame:
    calibration = pairs[pairs["Season"] == "E2023"].copy()
    tau, alpha, _, grid, _, _ = engine.select_hyperparameters(calibration)
    raw_group, audit = engine.fit_pairs(calibration, alpha, tau)
    raw = engine.EXPANSION @ raw_group
    gamma, rms = engine.rms_anchor(calibration, raw)
    final = gamma * raw
    rows: list[dict[str, object]] = []
    for target in ("E2024", "E2025"):
        frame = pairs[pairs["Season"] == target]
        matrix = engine.design_matrix(frame)
        for estimand, official, proposed in (
            ("Non-PTS", matrix @ np.ones(10), matrix @ final),
            (
                "Full",
                frame["DeltaPTS"].to_numpy(float) + matrix @ np.ones(10),
                frame["DeltaPTS"].to_numpy(float) + matrix @ final,
            ),
        ):
            oc, rc = strict_counts(official), strict_counts(proposed)
            off_ok, rm_ok = official > TOLERANCE, proposed > TOLERANCE
            rows.append(
                {
                    "Role": "Retrospective fixed-origin sensitivity",
                    "CalibrationSeasons": "E2023",
                    "TargetSeason": target,
                    "Estimand": estimand,
                    "Games": len(frame),
                    "Tau": tau,
                    "Alpha": alpha,
                    "MaximumTuningConcordanceTies": int((grid["ValidationWinnerHigher"] == grid["ValidationWinnerHigher"].max()).sum()),
                    "Gamma": gamma,
                    "OfficialWinnerHigher": oc[0],
                    "OfficialTies": oc[1],
                    "OfficialLoserHigher": oc[2],
                    "RMWinnerHigher": rc[0],
                    "RMTies": rc[1],
                    "RMLoserHigher": rc[2],
                    "ReclassifiedToConcordant": int((~off_ok & rm_ok).sum()),
                    "NewlyDiscordant": int((off_ok & ~rm_ok).sum()),
                    "RMSIdentityError": rms["RMSIdentityError"],
                    "Converged": audit["Converged"],
                }
            )
    return pd.DataFrame(rows)


def location_scale(values: np.ndarray, method: str) -> float:
    values = np.asarray(values, dtype=float)
    if method == "SD":
        return float(np.std(values, ddof=0))
    if method == "MAD":
        return float(np.median(np.abs(values - np.median(values))))
    if method == "IQR":
        return float(np.quantile(values, 0.75) - np.quantile(values, 0.25))
    raise ValueError(method)


def anchor_factor(method: str, calibration: pd.DataFrame, totals: pd.DataFrame, raw: np.ndarray) -> float:
    matrix = engine.design_matrix(calibration)
    official = matrix @ np.ones(10)
    learned = matrix @ raw
    if method == "Game-contrast RMS":
        return float(np.sqrt(np.mean(official**2)) / np.sqrt(np.mean(learned**2)))
    if method == "Winner-oriented contrast SD":
        return location_scale(official, "SD") / location_scale(learned, "SD")
    if method.startswith("Team-total"):
        signed = totals[list(engine.COMPONENTS)].to_numpy(float) * engine.SIGNS
        official_team = signed @ np.ones(10)
        learned_team = signed @ raw
        scale_name = method.rsplit(" ", 1)[-1]
        return location_scale(official_team, scale_name) / location_scale(learned_team, scale_name)
    if method == "No normalization":
        return 1.0
    raise ValueError(method)


def scale_anchor_sensitivity(
    pairs: pd.DataFrame, totals: pd.DataFrame, models: pd.DataFrame, coefficients: pd.DataFrame
) -> pd.DataFrame:
    methods = (
        "Game-contrast RMS",
        "Winner-oriented contrast SD",
        "Team-total SD",
        "Team-total MAD",
        "Team-total IQR",
        "No normalization",
    )
    rows: list[dict[str, object]] = []
    for target, calibration_seasons in engine.EXPERIMENTS.items():
        calibration = pairs[pairs["Season"].isin(calibration_seasons)]
        calibration_totals = totals[totals["Season"].isin(calibration_seasons)]
        evaluation = pairs[pairs["Season"] == target]
        raw = (
            coefficients[coefficients["TargetSeason"] == target]
            .set_index("Component")
            .loc[list(engine.COMPONENTS), "RawMagnitude"]
            .to_numpy(float)
        )
        matrix = engine.design_matrix(evaluation)
        for method in methods:
            gamma = anchor_factor(method, calibration, calibration_totals, raw)
            final = gamma * raw
            for estimand, official, proposed in (
                ("Non-PTS", matrix @ np.ones(10), matrix @ final),
                (
                    "Full",
                    evaluation["DeltaPTS"].to_numpy(float) + matrix @ np.ones(10),
                    evaluation["DeltaPTS"].to_numpy(float) + matrix @ final,
                ),
            ):
                oc, rc = strict_counts(official), strict_counts(proposed)
                rows.append(
                    {
                        "TargetSeason": target,
                        "CalibrationSeasons": "+".join(calibration_seasons),
                        "Anchor": method,
                        "Gamma": gamma,
                        "Estimand": estimand,
                        "Games": len(evaluation),
                        "OfficialWinnerHigher": oc[0],
                        "RMWinnerHigher": rc[0],
                        "RMTies": rc[1],
                        "RMLoserHigher": rc[2],
                        "DiscordanceChangePP": 100 * ((rc[1] + rc[2]) - (oc[1] + oc[2])) / len(evaluation),
                        "TargetContrastMean": float(np.mean(proposed)),
                        "TargetContrastSD": float(np.std(proposed, ddof=0)),
                        "TargetContrastRMS": float(np.sqrt(np.mean(proposed**2))),
                        "FinalMagnitudeMinimum": float(final.min()),
                        "FinalMagnitudeMaximum": float(final.max()),
                    }
                )
    return pd.DataFrame(rows)


def player_scale_sensitivity(
    players: pd.DataFrame, pairs: pd.DataFrame, totals: pd.DataFrame, coefficients: pd.DataFrame
) -> pd.DataFrame:
    target = "E2025"
    calibration_seasons = engine.EXPERIMENTS[target]
    calibration = pairs[pairs["Season"].isin(calibration_seasons)]
    calibration_totals = totals[totals["Season"].isin(calibration_seasons)]
    raw = (
        coefficients[coefficients["TargetSeason"] == target]
        .set_index("Component")
        .loc[list(engine.COMPONENTS), "RawMagnitude"]
        .to_numpy(float)
    )
    sample = players[(players["Season"] == target) & (players["Minutes"] > 0)].copy()
    signed = sample[list(engine.COMPONENTS)].to_numpy(float) * engine.SIGNS
    baseline_scores: np.ndarray | None = None
    baseline_ranks: np.ndarray | None = None
    rows: list[dict[str, object]] = []
    methods = (
        "Game-contrast RMS",
        "Winner-oriented contrast SD",
        "Team-total SD",
        "Team-total MAD",
        "Team-total IQR",
        "No normalization",
    )
    for method in methods:
        gamma = anchor_factor(method, calibration, calibration_totals, raw)
        scores = sample["PTS"].to_numpy(float) + signed @ (gamma * raw)
        season = pd.DataFrame({"PlayerCode": sample["PlayerCode"], "Score": scores}).groupby("PlayerCode", as_index=False)["Score"].sum()
        ranks = season["Score"].rank(method="min", ascending=False).to_numpy(float)
        if baseline_scores is None:
            baseline_scores = season["Score"].to_numpy(float)
            baseline_ranks = ranks.copy()
        rows.append(
            {
                "Anchor": method,
                "Gamma": gamma,
                "PlayerGames": len(scores),
                "Mean": float(np.mean(scores)),
                "SD": float(np.std(scores, ddof=0)),
                "RMS": float(np.sqrt(np.mean(scores**2))),
                "IQR": float(np.quantile(scores, 0.75) - np.quantile(scores, 0.25)),
                "P1": float(np.quantile(scores, 0.01)),
                "P99": float(np.quantile(scores, 0.99)),
                "Minimum": float(np.min(scores)),
                "Maximum": float(np.max(scores)),
                "NegativeScores": int((scores < 0).sum()),
                "NegativeScorePercent": 100 * float((scores < 0).mean()),
                "SeasonRankSpearmanVsRMS": float(spearmanr(baseline_ranks, ranks).statistic),
                "Top20OverlapVsRMS": len(set(np.argsort(-baseline_scores)[:20]) & set(np.argsort(-season["Score"].to_numpy(float))[:20])),
            }
        )
    return pd.DataFrame(rows)


def tuning_tie_sensitivity(
    pairs: pd.DataFrame, tuning: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for target, calibration_seasons in engine.EXPERIMENTS.items():
        grid = tuning[tuning["TargetSeason"] == target]
        top = grid[grid["ValidationWinnerHigher"] == grid["ValidationWinnerHigher"].max()]
        calibration = pairs[pairs["Season"].isin(calibration_seasons)]
        evaluation = pairs[pairs["Season"] == target]
        matrix = engine.design_matrix(evaluation)
        for candidate in top.itertuples(index=False):
            raw_group, _ = engine.fit_pairs(calibration, float(candidate.Alpha), float(candidate.Tau))
            raw = engine.EXPANSION @ raw_group
            gamma, _ = engine.rms_anchor(calibration, raw)
            final = gamma * raw
            for estimand, official, proposed in (
                ("Non-PTS", matrix @ np.ones(10), matrix @ final),
                (
                    "Full",
                    evaluation["DeltaPTS"].to_numpy(float) + matrix @ np.ones(10),
                    evaluation["DeltaPTS"].to_numpy(float) + matrix @ final,
                ),
            ):
                oc, rc = strict_counts(official), strict_counts(proposed)
                rows.append(
                    {
                        "TargetSeason": target,
                        "CalibrationSeasons": "+".join(calibration_seasons),
                        "MaximumInnerConcordance": int(top["ValidationWinnerHigher"].max()),
                        "CandidatesTiedAtMaximum": len(top),
                        "Tau": float(candidate.Tau),
                        "Alpha": float(candidate.Alpha),
                        "SelectedByTieBreak": bool(candidate.Selected),
                        "Estimand": estimand,
                        "OfficialWinnerHigher": oc[0],
                        "RMWinnerHigher": rc[0],
                        "RMTies": rc[1],
                        "RMLoserHigher": rc[2],
                        "DiscordanceChangePP": 100 * ((rc[1] + rc[2]) - (oc[1] + oc[2])) / len(evaluation),
                    }
                )
    return pd.DataFrame(rows)


def club_membership_bootstrap(games: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, object]] = []
    for target in ("E2025", "Pooled"):
        frame = games if target == "Pooled" else games[games["Season"] == target]
        clubs = np.array(sorted(set(frame["Winner"]) | set(frame["Loser"])))
        club_index = {club: index for index, club in enumerate(clubs)}
        left = frame["Winner"].map(club_index).to_numpy(int)
        right = frame["Loser"].map(club_index).to_numpy(int)
        for estimand in ("Full", "Non-PTS"):
            official_column, rm_column = metric_columns(estimand)
            effect = 100 * (
                (frame[rm_column].to_numpy(float) <= TOLERANCE).astype(float)
                - (frame[official_column].to_numpy(float) <= TOLERANCE).astype(float)
            )
            draws = np.empty(CLUB_BOOTSTRAPS)
            for b in range(CLUB_BOOTSTRAPS):
                multiplicity = np.bincount(rng.integers(0, len(clubs), len(clubs)), minlength=len(clubs))
                weight = multiplicity[left] * multiplicity[right]
                draws[b] = np.average(effect, weights=weight) if weight.sum() else np.nan
            draws = draws[np.isfinite(draws)]
            rows.append(
                {
                    "TargetSeason": target,
                    "Estimand": estimand,
                    "Method": "Undirected club-membership pigeonhole bootstrap",
                    "Clubs": len(clubs),
                    "Games": len(frame),
                    "Repetitions": len(draws),
                    "ObservedDiscordanceChangePP": float(np.mean(effect)),
                    "Interval2_5PP": float(np.quantile(draws, 0.025)),
                    "Interval97_5PP": float(np.quantile(draws, 0.975)),
                    "ProbabilityChangeAtLeastZero": float(np.mean(draws >= 0)),
                }
            )
    return pd.DataFrame(rows)


def joint_uncertainty_bootstrap(
    pairs: pd.DataFrame, model: pd.Series
) -> pd.DataFrame:
    rng = np.random.default_rng(SEED + 1)
    calibration_seasons = str(model["CalibrationSeasons"]).split("+")
    calibration = pairs[pairs["Season"].isin(calibration_seasons)].reset_index(drop=True)
    target = pairs[pairs["Season"] == "E2025"].reset_index(drop=True)
    target_matrix = engine.design_matrix(target)
    official_np = target_matrix @ np.ones(10)
    official_full = target["DeltaPTS"].to_numpy(float) + official_np
    clubs = np.array(sorted(set(target["Winner"]) | set(target["Loser"])))
    club_index = {club: index for index, club in enumerate(clubs)}
    left = target["Winner"].map(club_index).to_numpy(int)
    right = target["Loser"].map(club_index).to_numpy(int)
    draws = {"Full": np.empty(JOINT_BOOTSTRAPS), "Non-PTS": np.empty(JOINT_BOOTSTRAPS)}
    for b in range(JOINT_BOOTSTRAPS):
        sampled = calibration.iloc[rng.integers(0, len(calibration), len(calibration))].copy()
        raw_group, _ = engine.fit_pairs(sampled, float(model["Alpha"]), float(model["Tau"]))
        raw = engine.EXPANSION @ raw_group
        gamma, _ = engine.rms_anchor(sampled, raw)
        proposed_np = target_matrix @ (gamma * raw)
        proposed_full = target["DeltaPTS"].to_numpy(float) + proposed_np
        multiplicity = np.bincount(rng.integers(0, len(clubs), len(clubs)), minlength=len(clubs))
        weight = multiplicity[left] * multiplicity[right]
        for estimand, official, proposed in (
            ("Full", official_full, proposed_full),
            ("Non-PTS", official_np, proposed_np),
        ):
            effect = 100 * ((proposed <= TOLERANCE).astype(float) - (official <= TOLERANCE).astype(float))
            draws[estimand][b] = np.average(effect, weights=weight) if weight.sum() else np.nan
    rows: list[dict[str, object]] = []
    for estimand, values in draws.items():
        values = values[np.isfinite(values)]
        rows.append(
            {
                "TargetSeason": "E2025",
                "Estimand": estimand,
                "Method": "Calibration-game refit plus target club-membership bootstrap",
                "HyperparameterTreatment": "Selected tau and alpha fixed",
                "CalibrationGames": len(calibration),
                "TargetGames": len(target),
                "Repetitions": len(values),
                "Interval2_5PP": float(np.quantile(values, 0.025)),
                "MedianPP": float(np.quantile(values, 0.5)),
                "Interval97_5PP": float(np.quantile(values, 0.975)),
                "ProbabilityChangeAtLeastZero": float(np.mean(values >= 0)),
            }
        )
    return pd.DataFrame(rows)


def partial_spearman(x: np.ndarray, y: np.ndarray, covariate: np.ndarray) -> float:
    x_rank, y_rank, z_rank = rankdata(x), rankdata(y), rankdata(covariate)
    design = np.column_stack((np.ones(len(z_rank)), z_rank))
    x_residual = x_rank - design @ np.linalg.lstsq(design, x_rank, rcond=None)[0]
    y_residual = y_rank - design @ np.linalg.lstsq(design, y_rank, rcond=None)[0]
    return float(np.corrcoef(x_residual, y_residual)[0, 1])


def association_statistics(frame: pd.DataFrame) -> dict[str, float]:
    eligible = frame[frame["PlayerPossessions"] >= 18]
    return {
        "OfficialRawSpearmanPlusMinus": float(spearmanr(frame["PIR"], frame["PlusMinus"]).statistic),
        "RMRawSpearmanPlusMinus": float(spearmanr(frame["RMPIR"], frame["PlusMinus"]).statistic),
        "OfficialPartialSpearmanMinutes": partial_spearman(frame["PIR"].to_numpy(float), frame["PlusMinus"].to_numpy(float), frame["Minutes"].to_numpy(float)),
        "RMPartialSpearmanMinutes": partial_spearman(frame["RMPIR"].to_numpy(float), frame["PlusMinus"].to_numpy(float), frame["Minutes"].to_numpy(float)),
        "OfficialPartialSpearmanPossessions": partial_spearman(frame["PIR"].to_numpy(float), frame["PlusMinus"].to_numpy(float), frame["PlayerPossessions"].to_numpy(float)),
        "RMPartialSpearmanPossessions": partial_spearman(frame["RMPIR"].to_numpy(float), frame["PlusMinus"].to_numpy(float), frame["PlayerPossessions"].to_numpy(float)),
        "OfficialSharedDenominatorSpearman": float(spearmanr(eligible["PIR100"], eligible["U100"]).statistic),
        "RMSharedDenominatorSpearman": float(spearmanr(eligible["RMPIR100"], eligible["U100"]).statistic),
    }


def player_association_diagnostics(
    players: pd.DataFrame, totals: pd.DataFrame, coefficients: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    magnitudes = (
        coefficients[coefficients["TargetSeason"] == "E2025"]
        .set_index("Component")
        .loc[list(engine.COMPONENTS), "FinalMagnitude"]
        .to_numpy(float)
    )
    sample = players[(players["Season"] == "E2025") & (players["Minutes"] > 0)].copy()
    signed = sample[list(engine.COMPONENTS)].to_numpy(float) * engine.SIGNS
    sample["RMPIR"] = sample["PTS"].to_numpy(float) + signed @ magnitudes

    team = totals[totals["Season"] == "E2025"].copy()
    opponent = team[["Season", "GameCode", "TeamCode", "PTS"]].rename(
        columns={"TeamCode": "OpponentCode", "PTS": "OpponentPTS"}
    )
    team = team.merge(opponent, on=["Season", "GameCode", "OpponentCode"], validate="one_to_one")
    team["TeamMargin"] = team["PTS"] - team["OpponentPTS"]
    coefficient = 6532 / 15807
    team["OneTeamPossessions"] = team["FGA"] + coefficient * team["FTA"] - team["OREB"] + team["TO"]
    balanced = team.groupby(["Season", "GameCode"], as_index=False)["OneTeamPossessions"].mean().rename(columns={"OneTeamPossessions": "GamePossessions"})
    sample = sample.merge(team[["Season", "GameCode", "TeamCode", "TeamMargin"]], on=["Season", "GameCode", "TeamCode"], validate="many_to_one")
    sample = sample.merge(balanced, on=["Season", "GameCode"], validate="many_to_one")
    sample["PlayerPossessions"] = sample["GamePossessions"] * sample["Minutes"] / sample["GameMinutes"]
    sample["TeamMarginResidualizedPM"] = sample["PlusMinus"] - sample["Minutes"] / sample["GameMinutes"] * sample["TeamMargin"]
    sample["PIR100"] = 100 * sample["PIR"] / sample["PlayerPossessions"]
    sample["RMPIR100"] = 100 * sample["RMPIR"] / sample["PlayerPossessions"]
    sample["U100"] = 100 * sample["TeamMarginResidualizedPM"] / sample["PlayerPossessions"]

    stats = association_statistics(sample)
    rows = [
        {
            "Season": "E2025",
            "PositiveMinutePlayerGames": len(sample),
            "PossessionThreshold": 18,
            "ThresholdEligiblePlayerGames": int((sample["PlayerPossessions"] >= 18).sum()),
            **stats,
            "RawSpearmanDifference": stats["RMRawSpearmanPlusMinus"] - stats["OfficialRawSpearmanPlusMinus"],
            "PartialMinutesDifference": stats["RMPartialSpearmanMinutes"] - stats["OfficialPartialSpearmanMinutes"],
            "PartialPossessionsDifference": stats["RMPartialSpearmanPossessions"] - stats["OfficialPartialSpearmanPossessions"],
            "SharedDenominatorDifference": stats["RMSharedDenominatorSpearman"] - stats["OfficialSharedDenominatorSpearman"],
        }
    ]

    # Subsequent-appearance diagnostics avoid a same-row shared denominator.
    sample = sample.sort_values(["PlayerCode", "Date", "GameCode"]).copy()
    future_rows: list[dict[str, float]] = []
    for _, group in sample.groupby("PlayerCode", sort=False):
        values = group["TeamMarginResidualizedPM"].to_numpy(float)
        indices = group.index.to_numpy()
        for position, index in enumerate(indices):
            future_rows.append(
                {
                    "Index": index,
                    "Future1": values[position + 1] if position + 1 < len(values) else np.nan,
                    "Future3": float(np.mean(values[position + 1 : position + 4])) if position + 3 < len(values) else np.nan,
                }
            )
    future = pd.DataFrame(future_rows).set_index("Index")
    sample = sample.join(future)
    for appearances, column in ((1, "Future1"), (3, "Future3")):
        eligible = sample[sample[column].notna()]
        rows.append(
            {
                "Season": f"E2025 next-{appearances}-appearance",
                "PositiveMinutePlayerGames": len(eligible),
                "PossessionThreshold": np.nan,
                "ThresholdEligiblePlayerGames": np.nan,
                "OfficialRawSpearmanPlusMinus": float(spearmanr(eligible["PIR"], eligible[column]).statistic),
                "RMRawSpearmanPlusMinus": float(spearmanr(eligible["RMPIR"], eligible[column]).statistic),
                "RawSpearmanDifference": float(spearmanr(eligible["RMPIR"], eligible[column]).statistic - spearmanr(eligible["PIR"], eligible[column]).statistic),
            }
        )

    # Whole-game cluster resampling keeps all players from a game together.
    rng = np.random.default_rng(SEED + 2)
    groups = [frame for _, frame in sample.groupby("GameCode", sort=False)]
    draws = {key: np.empty(PLAYER_BOOTSTRAPS) for key in ("Raw", "PartialMinutes", "PartialPossessions", "SharedDenominator")}
    for b in range(PLAYER_BOOTSTRAPS):
        frame = pd.concat([groups[index] for index in rng.integers(0, len(groups), len(groups))], ignore_index=True)
        values = association_statistics(frame)
        draws["Raw"][b] = values["RMRawSpearmanPlusMinus"] - values["OfficialRawSpearmanPlusMinus"]
        draws["PartialMinutes"][b] = values["RMPartialSpearmanMinutes"] - values["OfficialPartialSpearmanMinutes"]
        draws["PartialPossessions"][b] = values["RMPartialSpearmanPossessions"] - values["OfficialPartialSpearmanPossessions"]
        draws["SharedDenominator"][b] = values["RMSharedDenominatorSpearman"] - values["OfficialSharedDenominatorSpearman"]
    bootstrap = pd.DataFrame(
        [
            {
                "Diagnostic": key,
                "Cluster": "GameCode",
                "Repetitions": PLAYER_BOOTSTRAPS,
                "Difference2_5": float(np.quantile(values, 0.025)),
                "DifferenceMedian": float(np.quantile(values, 0.5)),
                "Difference97_5": float(np.quantile(values, 0.975)),
            }
            for key, values in draws.items()
        ]
    )
    return pd.DataFrame(rows), bootstrap


def exclusions_audit() -> pd.DataFrame:
    result = pd.read_csv(engine.SOURCE_EXCLUSIONS)
    result = result.sort_values(["Season", "GameCode"]).reset_index(drop=True)
    assert len(result[result["Season"] == "E2021"]) == 28
    return result


def provenance_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("Winner-higher outcome criterion", "Exploratory project phase before the present review", "E2023--E2025", "Post hoc"),
            ("Eight action groups and equality restrictions", "Exploratory project phase before the present review", "E2023--E2025", "Post hoc"),
            ("Raw bounds [0.5,1.5]", "Exploratory project phase before the present review", "E2023--E2025", "Post hoc"),
            ("Calibration-window RMS match", "Exploratory project phase before the present review", "E2023--E2025", "Post hoc"),
            ("Tau/alpha grid and lexicographic tie-break", "Supervisor-revision analysis, July 2026", "E2020--E2025", "Post hoc"),
            ("Three-season rolling operational rule", "Supervisor-revision analysis, July 2026", "E2020--E2025", "Post hoc historical backtest"),
            ("E2025 latest-season focus", "Chosen as the latest complete season after inspection", "E2023--E2025", "Post hoc"),
            ("Tie, scale, dependence, and player diagnostics", "Technical revision, August 2026", "E2020--E2025", "Post hoc sensitivity"),
        ],
        columns=("Decision", "WhenMade", "SeasonsAlreadyInspected", "Classification"),
    )


def figure_methodology() -> None:
    fig, ax = plt.subplots(figsize=(10.2, 3.2))
    ax.set_xlim(0, 10.2)
    ax.set_ylim(0, 3.2)
    ax.axis("off")
    boxes = [
        (0.15, 1.15, 1.55, 1.15, "Completed prior\nteam box scores"),
        (2.05, 1.15, 1.65, 1.15, "Winner--loser\nnon-PTS contrasts"),
        (4.05, 1.15, 1.75, 1.15, "Inner tuning and\nfull-window refit"),
        (6.15, 1.15, 1.55, 1.15, "Calibration-window\nRMS match"),
        (8.05, 1.15, 1.85, 1.15, "Season-locked RM-PIR\nfor target rows"),
    ]
    for index, (x, y, width, height, label) in enumerate(boxes):
        ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.04,rounding_size=0.07", facecolor="#eaf4fb" if index < 4 else "#d5e9f6", edgecolor="#2171b5", linewidth=1.2))
        ax.text(x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=9)
    for index in range(len(boxes) - 1):
        x1, x2 = boxes[index][0] + boxes[index][2], boxes[index + 1][0]
        ax.add_patch(FancyArrowPatch((x1 + 0.04, 1.72), (x2 - 0.04, 1.72), arrowstyle="-|>", mutation_scale=11, color="#4d4d4d", linewidth=1.0))
    ax.text(4.92, 2.75, r"Fitted criterion: $z_g(\widehat{\mathbf{a}})>0$", ha="center", fontsize=10, weight="bold")
    ax.text(8.97, 0.66, r"$\Delta_g^{\mathrm{RM}}=\Delta_g^{\mathrm{PTS}}+\gamma z_g(\widehat{\mathbf{a}})$", ha="center", fontsize=10)
    ax.annotate("PTS coefficient fixed at 1; PTS absent from fitting", xy=(8.15, 1.15), xytext=(6.1, 0.25), arrowprops={"arrowstyle": "->", "color": "#d95f0e"}, color="#a63603", fontsize=8.5)
    fig.savefig(FIGURES / "vppir_methodology.pdf", metadata={"Title": "Season-locked RM-PIR calibration protocol"})
    plt.close(fig)


def figure_validation(validation: pd.DataFrame) -> None:
    focal = validation[validation["TargetSeason"] == "E2025"].set_index("Estimand")
    labels = ["Complete score\n(primary operational)", "Non-PTS\n(mechanism check)"]
    official = [focal.loc["Full", "OfficialErrorRatePercent"], focal.loc["Non-PTS", "OfficialErrorRatePercent"]]
    proposed = [focal.loc["Full", "VPErrorRatePercent"], focal.loc["Non-PTS", "VPErrorRatePercent"]]
    x = np.arange(2)
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    ax.bar(x - 0.19, official, 0.38, label="Official PIR", color="#9ecae1")
    ax.bar(x + 0.19, proposed, 0.38, label="RM-PIR", color="#2171b5")
    for positions, values in ((x - 0.19, official), (x + 0.19, proposed)):
        for position, value in zip(positions, values):
            ax.text(position, value + 0.25, f"{value:.2f}%", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Discordance rate (% of 2025–26 games)")
    ax.set_ylim(0, max(official + proposed) * 1.25)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    ax.grid(axis="y", color="#ececec", linewidth=0.6)
    fig.savefig(FIGURES / "vppir_validation.pdf", metadata={"Title": "2025–26 RM-PIR discordance"})
    plt.close(fig)


def figure_changed_games(games: pd.DataFrame) -> None:
    focal = games[games["Season"] == "E2025"].copy()
    gain = (focal["OfficialFullDiff"] <= TOLERANCE) & (focal["VPFullDiff"] > TOLERANCE)
    loss = (focal["OfficialFullDiff"] > TOLERANCE) & (focal["VPFullDiff"] <= TOLERANCE)
    assert (int(gain.sum()), int(loss.sum())) == (9, 3)
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    ax.scatter(focal["OfficialFullDiff"], focal["VPFullDiff"], s=18, color="#9ECAE1", alpha=0.70, edgecolors="none", zorder=1, label="All games")
    ax.scatter(focal.loc[gain, "OfficialFullDiff"], focal.loc[gain, "VPFullDiff"], s=34, color="#2171B5", edgecolors="white", linewidths=0.45, zorder=2, label="Reclassified concordant (n=9)")
    ax.scatter(focal.loc[loss, "OfficialFullDiff"], focal.loc[loss, "VPFullDiff"], s=38, marker="D", color="#D95F0E", edgecolors="white", linewidths=0.45, zorder=3, label="New RM-PIR discordance (n=3)")
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
        color, linewidth, alpha = ("#2171b5", 1.8, 0.95) if row.RankChange > 0 else (("#d95f0e", 1.2, 0.75) if row.RankChange < 0 else ("#9e9e9e", 0.9, 0.65))
        ax.plot([0, 1], [row.OfficialRank, row.VPRank], color=color, linewidth=linewidth, alpha=alpha)
        ax.scatter([0, 1], [row.OfficialRank, row.VPRank], color=color, s=18, zorder=2)
        ax.text(-0.03, row.OfficialRank, f"{row.OfficialRank:>2}  {row.TeamCode}", ha="right", va="center", fontsize=8)
        suffix = f" ({int(row.RankChange):+d})" if row.RankChange else ""
        ax.text(1.03, row.VPRank, f"{row.TeamCode}  {row.VPRank:>2}{suffix}", ha="left", va="center", fontsize=8)
    ax.text(0, 0.3, "Official PIR rank", ha="center", va="bottom", fontsize=10, weight="bold")
    ax.text(1, 0.3, "RM-PIR rank", ha="center", va="bottom", fontsize=10, weight="bold")
    ax.text(0.5, len(teams) + 0.55, "Positive parenthetical values denote upward movement", ha="center", fontsize=7.5, color="#555555")
    fig.savefig(FIGURES / "vppir_team_rank_movement.pdf", metadata={"Title": "E2025 regular-season RM-PIR team-rank movement"})
    plt.close(fig)


def figure_coefficients(coefficients: pd.DataFrame, intervals: pd.DataFrame) -> None:
    targets = list(engine.EXPERIMENTS)
    matrix = np.empty((len(engine.COMPONENTS), len(targets)))
    low = np.empty_like(matrix)
    high = np.empty_like(matrix)
    boundary = np.zeros_like(matrix, dtype=bool)
    for column, target in enumerate(targets):
        frame = coefficients[coefficients["TargetSeason"] == target].set_index("Component").loc[list(engine.COMPONENTS)]
        interval = intervals[intervals["TargetSeason"] == target].set_index("Component").loc[list(engine.COMPONENTS)]
        matrix[:, column] = frame["FinalMagnitude"]
        signed_low, signed_high = interval["SignedCI2_5"].to_numpy(float), interval["SignedCI97_5"].to_numpy(float)
        low[:, column], high[:, column] = np.minimum(np.abs(signed_low), np.abs(signed_high)), np.maximum(np.abs(signed_low), np.abs(signed_high))
        boundary[:, column] = frame["RawAtLowerBound"].to_numpy(bool) | frame["RawAtUpperBound"].to_numpy(bool)
    x, width = np.arange(len(engine.COMPONENTS), dtype=float), 0.145
    colors = ("#0072B2", "#56B4E9", "#009E73", "#E69F00", "#CC79A7")
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.axvspan(-0.55, 4.5, color="#eaf4fb", alpha=0.65, zorder=0)
    ax.axvspan(4.5, 9.55, color="#fff2e2", alpha=0.65, zorder=0)
    for column, (target, color) in enumerate(zip(targets, colors)):
        positions = x + (column - 2) * width
        ax.bar(positions, matrix[:, column], width=0.92 * width, color=color, edgecolor="white", linewidth=0.35, yerr=np.vstack((matrix[:, column] - low[:, column], high[:, column] - matrix[:, column])), error_kw={"ecolor": "#333333", "elinewidth": 0.55, "capsize": 1.2, "capthick": 0.55}, label=engine.SEASON_LABELS[target], zorder=2)
        active = boundary[:, column]
        ax.scatter(positions[active], matrix[active, column] + 0.045, s=18, facecolors="none", edgecolors="black", linewidths=0.75, zorder=5)
    ax.axhline(1.0, color="#222222", linewidth=1.15, linestyle=(0, (5, 3)), zorder=4)
    ax.axvline(4.5, color="#555555", linewidth=1.0, zorder=4)
    ax.text(9.48, 1.035, "Official PIR weight", ha="right", va="bottom", fontsize=8, color="#222222", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.5}, zorder=6)
    ax.text(2.0, 1.025, "Positive action credits", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=9, weight="bold", color="#1f5f89")
    ax.text(7.0, 1.025, "Penalty components", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=9, weight="bold", color="#9a5b13")
    ax.set_xticks(x, engine.COMPONENTS)
    ax.set_xlim(-0.55, len(engine.COMPONENTS) - 0.45)
    ax.set_ylim(0, float(high.max()) * 1.17)
    ax.set_ylabel(r"Final RM-PIR coefficient magnitude, $|c_{j,Y}|$")
    ax.set_xlabel("PIR component")
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.55, zorder=1)
    ax.legend(title="Target season", frameon=False, ncol=len(targets), loc="upper center", bbox_to_anchor=(0.5, 1.19), columnspacing=1.3, handlelength=1.3, fontsize=8, title_fontsize=8)
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.14, top=0.77)
    fig.savefig(FIGURES / "vppir_coefficient_stability.pdf", metadata={"Title": "Season-locked RM-PIR coefficient magnitudes"})
    plt.close(fig)


def figure_historical(validation: pd.DataFrame) -> None:
    seasons = list(engine.EXPERIMENTS)
    labels = [engine.SEASON_LABELS[season] for season in seasons]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8), gridspec_kw={"width_ratios": [1.15, 1.0]})
    colors = {"Non-PTS": "#2171b5", "Full": "#d95f0e"}
    for estimand in ("Non-PTS", "Full"):
        frame = validation[(validation["Estimand"] == estimand) & (validation["TargetSeason"] != "Pooled")].set_index("TargetSeason").loc[seasons]
        axes[0].plot(labels, frame["OfficialErrorRatePercent"], marker="o", linestyle="--", color=colors[estimand], alpha=0.55, label=f"Official {estimand}")
        axes[0].plot(labels, frame["VPErrorRatePercent"], marker="s", linestyle="-", color=colors[estimand], label=f"RM-PIR {estimand}")
        axes[1].bar(np.arange(len(seasons)) + (-0.17 if estimand == "Non-PTS" else 0.17), frame["ErrorRateDifferencePP"], width=0.34, color=colors[estimand], label=estimand)
    axes[0].set_title("a  Target-season discordance rate")
    axes[0].set_ylabel("Discordance (% of games)")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].grid(axis="y", color="#ececec", linewidth=0.6)
    axes[0].legend(frameon=False, fontsize=7, ncol=2)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xticks(np.arange(len(seasons)), labels, rotation=30)
    axes[1].set_title("b  Signed discordance-rate change")
    axes[1].set_ylabel("RM-PIR minus official PIR (percentage points)")
    axes[1].legend(frameon=False, fontsize=7)
    axes[1].grid(axis="y", color="#ececec", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(FIGURES / "vppir_historical_validation.pdf", metadata={"Title": "Historical rolling-origin RM-PIR backtest"})
    plt.close(fig)


def update_manifest(outputs: list[Path]) -> None:
    path = DERIVED / "analysis_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["analysis"] = "Season-locked RMS-matched PIR rolling backtest and technical-revision audits"
    manifest["technical_revision_seed"] = SEED
    manifest["club_membership_bootstraps"] = CLUB_BOOTSTRAPS
    manifest["joint_uncertainty_bootstraps"] = JOINT_BOOTSTRAPS
    manifest["player_diagnostic_bootstraps"] = PLAYER_BOOTSTRAPS
    manifest["common_resolution_equivalence_band"] = EQUIVALENCE_BAND
    for output in outputs:
        manifest["outputs"][output.relative_to(ROOT).as_posix()] = sha256(output)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    metadata = engine.load_metadata()
    totals = engine.load_team_totals(metadata)
    players = engine.load_players()
    pairs = engine.build_pairs(totals)
    validation = pd.read_csv(DERIVED / "rolling_validation_summary.csv")
    games = pd.read_csv(DERIVED / "rolling_game_results.csv")
    models = pd.read_csv(DERIVED / "model_registry.csv")
    coefficients = pd.read_csv(DERIVED / "rolling_coefficients.csv")
    tuning = pd.read_csv(DERIVED / "nested_tuning_grid.csv")
    intervals = pd.read_csv(DERIVED / "coefficient_bootstrap_intervals.csv")

    outputs: dict[str, pd.DataFrame] = {
        "tie_sensitivity.csv": tie_sensitivity(games),
        "figure_category_audit.csv": figure_category_audit(games),
        "fixed_origin_transport.csv": fixed_origin_transport(pairs),
        "scale_anchor_sensitivity.csv": scale_anchor_sensitivity(pairs, totals, models, coefficients),
        "player_scale_sensitivity.csv": player_scale_sensitivity(players, pairs, totals, coefficients),
        "tuning_tie_sensitivity.csv": tuning_tie_sensitivity(pairs, tuning),
        "club_membership_bootstrap.csv": club_membership_bootstrap(games),
        "joint_uncertainty_bootstrap.csv": joint_uncertainty_bootstrap(pairs, models[models["TargetSeason"] == "E2025"].iloc[0]),
        "data_exclusions.csv": exclusions_audit(),
        "analysis_provenance.csv": provenance_table(),
    }
    player_diagnostics, player_bootstrap = player_association_diagnostics(players, totals, coefficients)
    outputs["player_association_diagnostics.csv"] = player_diagnostics
    outputs["player_association_bootstrap.csv"] = player_bootstrap
    written: list[Path] = []
    for filename, table in outputs.items():
        path = DERIVED / filename
        table.to_csv(path, index=False)
        written.append(path)

    figure_methodology()
    figure_validation(validation)
    figure_changed_games(games)
    figure_team_ranks(pd.read_csv(DERIVED / "e2025_teams_regular_season.csv"))
    figure_coefficients(coefficients, intervals)
    figure_historical(validation)
    figure_paths = [
        FIGURES / "vppir_methodology.pdf",
        FIGURES / "vppir_validation.pdf",
        FIGURES / "vppir_changed_games.pdf",
        FIGURES / "vppir_team_rank_movement.pdf",
        FIGURES / "vppir_coefficient_stability.pdf",
        FIGURES / "vppir_historical_validation.pdf",
    ]
    update_manifest(written + figure_paths)
    print("Wrote technical-revision audits:", ", ".join(outputs))
    print(outputs["figure_category_audit.csv"].query("TargetSeason == 'E2025'").to_string(index=False))
    print(outputs["club_membership_bootstrap.csv"].to_string(index=False))
    print(outputs["joint_uncertainty_bootstrap.csv"].to_string(index=False))
    print(player_diagnostics.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
