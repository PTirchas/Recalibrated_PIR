"""Fast integrity tests for generated supervisor-revision artifacts."""

from __future__ import annotations

import hashlib
import json
import re
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
HERE = ROOT / "manuscript"
DERIVED = ROOT / "tables"


class RevisionArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.models = pd.read_csv(DERIVED / "model_registry.csv")
        cls.validation = pd.read_csv(DERIVED / "rolling_validation_summary.csv")
        cls.players = pd.read_csv(DERIVED / "e2025_players.csv")
        cls.teams = pd.read_csv(DERIVED / "e2025_teams_regular_season.csv")
        cls.teams_all_stages = pd.read_csv(DERIVED / "e2025_teams_all_stages.csv")

    def test_expected_coverage(self) -> None:
        audit = pd.read_csv(DERIVED / "data_integrity_audit.csv")
        observed = dict(zip(audit["Season"], audit["IncludedGames"]))
        self.assertEqual(
            observed,
            {"E2020": 328, "E2021": 299, "E2022": 328, "E2023": 331, "E2024": 330, "E2025": 402},
        )
        self.assertEqual(int(audit["IncludedGames"].sum()), 2018)
        self.assertEqual(int(audit["OfficialPIRReconstructionMismatches"].sum()), 0)

    def test_temporal_leakage_absent(self) -> None:
        order = {season: index for index, season in enumerate(("E2020", "E2021", "E2022", "E2023", "E2024", "E2025"))}
        for row in self.models.itertuples(index=False):
            calibration = str(row.CalibrationSeasons).split("+")
            self.assertNotIn(row.TargetSeason, calibration)
            self.assertTrue(all(order[season] < order[row.TargetSeason] for season in calibration))

    def test_rms_identity_and_final_bounds(self) -> None:
        self.assertLess(float(self.models["RMSIdentityError"].max()), 1e-10)
        self.assertTrue((self.models["FinalMagnitudeMinimum"] >= self.models["Gamma"] * 0.5 - 1e-10).all())
        self.assertTrue((self.models["FinalMagnitudeMaximum"] <= self.models["Gamma"] * 1.5 + 1e-10).all())

    def test_validation_partitions_and_pooled_totals(self) -> None:
        for row in self.validation.itertuples(index=False):
            self.assertEqual(row.OfficialWinnerHigher + row.OfficialTies + row.OfficialLoserHigher, row.Games)
            self.assertEqual(row.VPWinnerHigher + row.VPTies + row.VPLoserHigher, row.Games)
            self.assertLessEqual(row.OfficialErrorRateCI2_5Percent, row.OfficialErrorRatePercent)
            self.assertGreaterEqual(row.OfficialErrorRateCI97_5Percent, row.OfficialErrorRatePercent)
            self.assertLessEqual(row.VPErrorRateCI2_5Percent, row.VPErrorRatePercent)
            self.assertGreaterEqual(row.VPErrorRateCI97_5Percent, row.VPErrorRatePercent)
        pooled = self.validation[self.validation["TargetSeason"] == "Pooled"]
        self.assertTrue(pooled["Games"].eq(1690).all())

        transitions = pd.read_csv(DERIVED / "paired_transitions.csv")
        merged = self.validation.merge(
            transitions,
            on=["Experiment", "TargetSeason", "Estimand"],
            validate="one_to_one",
        )
        self.assertTrue(
            (
                merged[["BothSuccess", "OfficialOnlySuccess", "VPOnlySuccess", "BothFailure"]]
                .sum(axis=1)
                == merged["Games"]
            ).all()
        )
        self.assertTrue((merged["OfficialOnlySuccess"] == merged["NewFailures"]).all())
        self.assertTrue((merged["VPOnlySuccess"] == merged["Corrections"]).all())

    def test_player_table_population_and_ranks(self) -> None:
        self.assertEqual(len(self.players), 335)
        expected_official = self.players["OfficialPIR"].rank(method="min", ascending=False).astype(int)
        expected_vp = self.players["VPPIR"].rank(method="min", ascending=False).astype(int)
        self.assertTrue(expected_official.equals(self.players["OfficialRank"]))
        self.assertTrue(expected_vp.equals(self.players["VPRank"]))
        self.assertTrue((self.players["RankChange"] == self.players["OfficialRank"] - self.players["VPRank"]).all())

    def test_team_table_population_common_stage_and_ranks(self) -> None:
        self.assertEqual(len(self.teams), 20)
        self.assertTrue(self.teams["GP"].eq(38).all())
        expected_official = self.teams["OfficialPIRPerGame"].rank(method="min", ascending=False).astype(int)
        expected_vp = self.teams["VPPIRPerGame"].rank(method="min", ascending=False).astype(int)
        self.assertTrue(expected_official.equals(self.teams["OfficialRank"]))
        self.assertTrue(expected_vp.equals(self.teams["VPRank"]))

    def test_all_stage_team_table_population_and_ranks(self) -> None:
        teams = self.teams_all_stages
        self.assertEqual(len(teams), 20)
        self.assertEqual((int(teams["GP"].min()), int(teams["GP"].max())), (38, 44))
        expected_official_total = teams["OfficialPIRTotal"].rank(
            method="min", ascending=False
        ).astype(int)
        expected_vp_total = teams["VPPIRTotal"].rank(
            method="min", ascending=False
        ).astype(int)
        self.assertTrue(expected_official_total.equals(teams["OfficialTotalRank"]))
        self.assertTrue(expected_vp_total.equals(teams["VPTotalRank"]))
        self.assertTrue(
            (
                teams["TotalRankChange"]
                == teams["OfficialTotalRank"] - teams["VPTotalRank"]
            ).all()
        )
        olympiacos = teams[teams["TeamCode"] == "OLY"].iloc[0]
        self.assertEqual(int(olympiacos["GP"]), 43)
        self.assertEqual(float(olympiacos["OfficialPIRTotal"]), 4727.0)
        self.assertAlmostEqual(float(olympiacos["OfficialPIRPerGame"]), 4727.0 / 43.0)
        self.assertAlmostEqual(float(olympiacos["VPPIRTotal"]), 4406.998809929648)

    def test_extended_sensitivity_and_bootstrap_outputs(self) -> None:
        sensitivity = pd.read_csv(DERIVED / "sensitivity_summary.csv")
        required = {
            "Expanded tau-alpha grid",
            "Multi-cutoff rolling-origin selection",
            "Centered-SD scale anchor",
            "Mean-absolute scale anchor",
            "Team-only residuals excluded",
            "Regular-season target only",
        }
        self.assertTrue(required.issubset(set(sensitivity["Analysis"])))
        nonpoint = sensitivity[
            (sensitivity["AnalysisRole"] == "Retrospective sensitivity")
            & (sensitivity["Estimand"] == "Non-PTS")
        ]
        self.assertTrue((nonpoint["ErrorDifferencePP"] < 0).all())

        dependence = pd.read_csv(DERIVED / "team_dependence_sensitivity.csv")
        self.assertEqual(len(dependence), 42)
        omissions = dependence[dependence["OmittedTeam"] != "None"]
        self.assertTrue((omissions["ErrorDifferencePP"] < 0).all())

        restrictions = pd.read_csv(DERIVED / "group_restriction_diagnostics.csv")
        focal = restrictions[restrictions["TargetSeason"] == "E2025"]
        self.assertEqual(set(focal["Pair"]), {"BLK-BLKA", "FD-FC"})
        self.assertTrue((focal["SignedContrastPearsonR"] > 0.98).all())

        intervals = pd.read_csv(DERIVED / "coefficient_bootstrap_intervals.csv")
        self.assertIn("RawLowerBoundaryFrequencyPercent", intervals.columns)
        self.assertIn("RawUpperBoundaryFrequencyPercent", intervals.columns)
        performance = pd.read_csv(DERIVED / "coefficient_bootstrap_target_performance.csv")
        self.assertEqual(len(performance), 10)
        self.assertTrue((performance["VPErrorRateCI2_5Percent"] <= performance["VPErrorRateMedianPercent"]).all())
        self.assertTrue((performance["VPErrorRateMedianPercent"] <= performance["VPErrorRateCI97_5Percent"]).all())

    def test_journal_structure_and_supplementary_transfer(self) -> None:
        manuscript = (HERE / "main.tex").read_text(encoding="utf-8")
        supplement = (HERE / "supplementary.tex").read_text(encoding="utf-8")
        self.assertIn(r"\documentclass[pdflatex,sn-mathphys-num]{sn-jnl}", manuscript)
        self.assertIn(r"\begin{abstract}", manuscript)
        self.assertIn(r"\end{abstract}", manuscript)
        self.assertLess(manuscript.index(r"\section{Methods}"), manuscript.index(r"\section{Results}"))
        self.assertLess(manuscript.index(r"\section{Results}"), manuscript.index(r"\section{Discussion}"))
        self.assertIn("The 2025--26 locked team coefficients were applied uniformly to player rows", supplement)
        self.assertIn("games played range from 38 to 44", supplement)
        self.assertIn("Olympiacos & 43 & 4727", supplement)
        self.assertIn("all 335 eligible player totals and ranks", supplement)
        self.assertNotIn(r"\begin{longtable}", manuscript)

    def test_technical_revision_audits(self) -> None:
        categories = pd.read_csv(DERIVED / "figure_category_audit.csv")
        focal = categories[categories["TargetSeason"] == "E2025"].set_index("Estimand")
        self.assertEqual(
            focal.loc["Full", ["BothConcordant", "OfficialOnlyConcordant", "RMOnlyConcordant", "BothDiscordant"]].astype(int).tolist(),
            [372, 3, 9, 18],
        )
        self.assertEqual(
            focal.loc["Non-PTS", ["BothConcordant", "OfficialOnlyConcordant", "RMOnlyConcordant", "BothDiscordant"]].astype(int).tolist(),
            [329, 5, 28, 40],
        )

        ties = pd.read_csv(DERIVED / "tie_sensitivity.csv")
        complete = ties[(ties["TargetSeason"] == "E2025") & (ties["Estimand"] == "Full")].set_index("Score")
        self.assertEqual(complete.loc["Official PIR", ["WinnerHigher", "Ties", "LoserHigher"]].astype(int).tolist(), [375, 3, 24])
        self.assertEqual(complete.loc["RM-PIR", ["WinnerHigher", "Ties", "LoserHigher"]].astype(int).tolist(), [381, 0, 21])
        self.assertEqual(complete.loc["RM-PIR", ["BandWinnerHigher", "BandTies", "BandLoserHigher"]].astype(int).tolist(), [378, 4, 20])

        scale = pd.read_csv(DERIVED / "scale_anchor_sensitivity.csv")
        focal_scale = scale[(scale["TargetSeason"] == "E2025") & (scale["Estimand"] == "Full")]
        self.assertEqual(focal_scale["Anchor"].nunique(), 6)
        self.assertEqual((int(focal_scale["RMWinnerHigher"].min()), int(focal_scale["RMWinnerHigher"].max())), (380, 382))
        focal_nonpoint = scale[(scale["TargetSeason"] == "E2025") & (scale["Estimand"] == "Non-PTS")]
        self.assertTrue(focal_nonpoint["RMWinnerHigher"].eq(357).all())

        associations = pd.read_csv(DERIVED / "player_association_diagnostics.csv").set_index("Season")
        self.assertEqual(int(associations.loc["E2025", "PositiveMinutePlayerGames"]), 8741)
        self.assertEqual(int(associations.loc["E2025 next-3-appearance", "PositiveMinutePlayerGames"]), 7764)

        exclusions = pd.read_csv(DERIVED / "data_exclusions.csv")
        self.assertEqual(len(exclusions[exclusions["Season"] == "E2021"]), 28)
        game_303 = exclusions[(exclusions["Season"] == "E2025") & (exclusions["GameCode"] == 303)]
        self.assertEqual(len(game_303), 1)

    def test_manuscript_technical_terminology(self) -> None:
        manuscript = (HERE / "main.tex").read_text(encoding="utf-8")
        self.assertNotIn("Variance-preserved", manuscript)
        self.assertNotIn("variance-preserved", manuscript)
        self.assertNotIn("paired failure indicator", manuscript)
        self.assertNotIn("ordering-error rate", manuscript)
        self.assertIn("Complete-score ordering was the primary operational estimand", manuscript)
        self.assertIn("uncentred second moments", manuscript)
        self.assertIn("algebraic decomposition, not causal attribution", manuscript)
        self.assertIn("Historical semantic invariance", manuscript)

    def test_optimizer_and_additivity_audits(self) -> None:
        optimizer = pd.read_csv(DERIVED / "optimization_audit.csv")
        self.assertTrue(optimizer["Passed"].all())
        residual = pd.read_csv(DERIVED / "residual_identity_audit.csv")
        self.assertEqual(int(residual["PTSResidualNonzeroTeamGames"].sum()), 0)
        self.assertLess(float(residual["VPAdditivityIdentityMaximumAbsoluteError"].dropna().max()), 1e-10)

    def test_main_display_limit(self) -> None:
        main = (HERE / "main.tex").read_text(encoding="utf-8")
        compiled = main.split(r"\end{document}", 1)[0]
        tables = len(re.findall(r"\\begin\{table\*?\}", compiled))
        figures = len(re.findall(r"\\begin\{figure\*?\}", compiled))
        self.assertEqual((tables, figures), (3, 5))
        self.assertLessEqual(tables + figures, 8)
        self.assertIn("figures/PIR_team_diagram.pdf", main)

    def test_historical_documentary_record(self) -> None:
        manuscript = (HERE / "main.tex").read_text(encoding="utf-8")
        evidence = (HERE / "historical_criteria_evidence.md").read_text(encoding="utf-8")
        bibliography = (HERE / "bibliography.bib").read_text(encoding="utf-8")
        for key in ("EuroleagueBylaws2021", "EuroleagueBylaws2022", "FIBAStatisticiansManual2024"):
            self.assertIn(key, manuscript)
            self.assertIn(f"{{{key},", bibliography)
        self.assertIn("fc898cbdbee1860e41f81490ca78c1c81609e3687e808dbdd7b34b14966985c5", evidence)
        self.assertIn("305d4aa613b6c6033f5e05b9f79c10aa27930be17191093a5930c304b1c6596d", evidence)

    def test_expanded_main_bibliography(self) -> None:
        manuscript = (HERE / "main.tex").read_text(encoding="utf-8")
        bibliography = (HERE / "bibliography.bib").read_text(encoding="utf-8")
        cited = {
            key.strip()
            for group in re.findall(r"\\cite\{([^}]+)\}", manuscript)
            for key in group.split(",")
        }
        available = set(re.findall(r"^@\w+\{([^,]+),", bibliography, flags=re.MULTILINE))
        self.assertTrue(cited.issubset(available))
        self.assertGreaterEqual(len(cited), 32)

    def test_vector_figures_and_manifest_hashes(self) -> None:
        expected = (
            "PIR_team_diagram.pdf",
            "vppir_methodology.pdf",
            "vppir_validation.pdf",
            "vppir_changed_games.pdf",
            "vppir_team_rank_movement.pdf",
            "vppir_coefficient_stability.pdf",
            "vppir_historical_validation.pdf",
        )
        for filename in expected:
            path = ROOT / "figures" / filename
            self.assertTrue(path.exists(), filename)
            self.assertEqual(path.read_bytes()[:4], b"%PDF")

        manifest = json.loads((DERIVED / "analysis_manifest.json").read_text(encoding="utf-8"))
        for relative, expected_hash in manifest["inputs"].items():
            path = ROOT / relative
            observed = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(observed, expected_hash, relative)
        for relative, expected_hash in manifest["outputs"].items():
            path = ROOT / relative
            observed = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(observed, expected_hash, relative)


if __name__ == "__main__":
    unittest.main()
