# Recalibrating EuroLeague Performance Index Rating for outcome-concordant team ledgers

Reproducibility repository for:

> **Recalibrating EuroLeague Performance Index Rating for outcome-concordant team ledgers: a season-locked historical evaluation**

**Panagiotis Tirchas, Nicholas Christakis, and Dimitris Drikakis**  
Institute for Advanced Modelling and Simulation, University of Nicosia, Nicosia, CY-2417, Cyprus

This repository contains the data-processing, model-fitting, robustness-analysis, validation, table-generation, and figure-generation code supporting the study. The scientific description and results below follow the current v3 main manuscript and Supplementary Information.

## Study overview

Official EuroLeague Performance Index Rating (PIR) is a transparent additive box-score ledger with equal unit magnitudes for its component actions. Equal weighting, however, does not always give the winning team a higher completed-game PIR total.

The study introduces **RMS-matched Performance Index Rating (RM-PIR)**, a constrained recalibration of the same official action ledger. RM-PIR:

- keeps the points coefficient fixed at `+1`;
- retains the official favourable and unfavourable action directions;
- estimates eight bounded nonpoint action-group magnitudes from completed prior seasons;
- matches the calibration-window root-mean-square scale of official nonpoint winner–loser contrasts;
- locks the resulting coefficients before scoring the target season; and
- applies the same target-season coefficients to teams and players without team- or player-specific parameters.

The primary operational outcome is **complete-score winner–loser concordance**. Nonpoint concordance is reported as the mechanism check because it evaluates the fitted action direction without the point margin that determines the winner.

## Headline findings

The analysis covers six EuroLeague seasons from 2020–21 through 2025–26:

| Quantity | Study population |
|---|---:|
| Completed games | 2,018 |
| Official team-game rows | 4,036 |
| Rolling target seasons | 5 |
| Rolling target games | 1,690 |
| Player-game rows across all six seasons | 43,549 |
| Positive-minute players in 2025–26 | 335 |

Across the five season-locked target seasons:

| Outcome | Official PIR discordances | RM-PIR discordances | Change |
|---|---:|---:|---:|
| Complete score | 150 | 104 | −2.72 percentage points |
| Nonpoint component | 293 | 209 | −4.97 percentage points |

For the 402 completed games in 2025–26:

| Outcome | Official PIR | RM-PIR | Paired change |
|---|---:|---:|---:|
| Complete score | 27 discordances | 21 discordances | −1.49 pp; game-bootstrap 95% interval [−3.23, 0.25] |
| Nonpoint component | 68 discordances | 45 discordances | −5.72 pp; game-bootstrap 95% interval [−8.46, −2.99] |

The 2025–26 complete-score comparison contained nine games reclassified into concordance and three newly introduced discordances. Its club-membership and joint coefficient/club intervals included zero. The nonpoint comparison contained 28 reclassified games and five new discordances; its club-membership and joint intervals remained negative. The paper therefore treats the complete-score result as modest and sensitivity-dependent, while the nonpoint mechanism change is larger and more consistent.

These are retrospective, criterion-specific historical results. They do not establish causal player value, predictive superiority, or context-complete individual impact.

## Evaluation design

Each target-season coefficient vector excludes that season from selection, fitting, and RMS matching:

| Target season | Calibration seasons |
|---|---|
| 2021–22 | 2020–21 |
| 2022–23 | 2020–21 to 2021–22 |
| 2023–24 | 2020–21 to 2022–23 |
| 2024–25 | 2021–22 to 2023–24 |
| 2025–26 | 2022–23 to 2024–25 |

Within each calibration window, the transition-width and ridge parameters are selected using a chronological 70/30 split of distinct dates. The chosen specification is refitted on the complete calibration window, RMS-matched there, and then locked before target-season evaluation.

The accompanying analyses include conditional coefficient bootstraps, paired whole-game bootstraps, club-membership resampling, joint calibration-refit/target-club resampling, tie-equivalence analyses, alternative scale anchors, alternative coefficient bounds and grouping assumptions, calibration-window changes, pace normalisation, overtime exclusion, withdrawn-club exclusion, regular-season-only evaluation, team-only residual checks, and secondary player-association diagnostics.

## Reproduction workflow

```text
Official EuroLeague game reports, box scores, and 2025–26 play-by-play
                                  |
                                  v
                     data/processed/*.csv
                                  |
                                  v
       prior-season selection, fitting, and RMS scale matching
                                  |
                     target-season coefficient lock
                                  |
                    +-------------+-------------+
                    |                           |
                    v                           v
             tables/*.csv                figures/*.pdf
                    |                           |
                    +-------------+-------------+
                                  v
                     manuscript checks and tests
```

## Quick start

Python 3.10 is the recorded analysis runtime.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m analytics.run_all --verify-only
```

`--verify-only` runs 16 integrity tests covering input and output hashes, season coverage, official PIR reconstruction, temporal separation, numerical identities, team/player ranks, reported counts, figure integrity, and manuscript structure.

Regenerate all analysis tables and the six programmatic RM-PIR figures from the bundled processed inputs:

```bash
python -m analytics.run_all
```

The deterministic analysis uses fixed random seeds and overwrites generated files under `tables/` and `figures/`.

To reacquire the public source data before rebuilding:

```bash
python -m analytics.run_all --refresh-data
```

This scans game codes 1–500 for all six seasons, retrieves played-game reports and box scores, and retrieves 2025–26 play-by-play. Responses are cached under the git-ignored `data/raw/` directory. Network availability or upstream API changes can affect a fresh acquisition; the bundled processed tables are the frozen study inputs.

## Repository contents

| Path | Purpose |
|---|---|
| `LICENSE` | GNU General Public License v3.0 licence text. |
| `analytics/collect_data.py` | Retrieves public EuroLeague reports, box scores, and play-by-play with local gzip caching and retry handling. |
| `analytics/build_datasets.py` | Converts cached responses into analysis-ready team and player tables. |
| `analytics/metrics.py` | Implements official PIR, possession estimates, and play-by-play helpers. |
| `analytics/rm_pir_analysis.py` | Runs the rolling-origin model, tuning, RMS matching, uncertainty analyses, integrity audits, tables, and primary figures. |
| `analytics/technical_robustness.py` | Runs tie, scale, dependence, fixed-origin, provenance, and player-association analyses. |
| `data/processed/` | Frozen official-source inputs used for the reported study. |
| `data/provenance/` | Original collection manifests for the historical and recent season groups. |
| `tables/` | Machine-readable primary, sensitivity, audit, ranking, and provenance outputs. |
| `figures/` | Vector manuscript figures. Legacy `vppir_` filenames are retained for stable manuscript references; the current terminology is RM-PIR. |
| `tests/` | Numerical, structural, temporal-leakage, and hash-integrity tests. |
| `manuscript/` | Current v3 manuscript PDFs plus manuscript-supporting LaTeX and bibliography files. |
| `docs/ANALYSIS_LINEAGE.md` | Identifies the final script path and explains which exploratory branches were superseded. |

Fixed seeds, numerical tolerances, bootstrap counts, dependency versions, and SHA-256 hashes are recorded in `tables/analysis_manifest.json`. See `tables/README.md` for the output map and `data/README.md` for source-data notes.

## Current manuscript files

- [Main manuscript — v3](manuscript/Nature_Scientific_Reports_PIR_Analysis_NC_v3_main.pdf)
- [Supplementary Information — v3](manuscript/Nature_Scientific_Reports_PIR_Analysis_NC_v3_Supplement.pdf)

The v3 main manuscript contains the complete study rationale, mathematical construction, rolling historical evaluation, primary results, discussion, availability statements, and author contributions. The Supplement contains extended coefficient-uncertainty methods, sensitivity analyses, team rankings, all-stage rankings, the complete 335-player table, additivity audits, and secondary player diagnostics.

The v3 PDFs are the authoritative scientific documents for this repository. The accompanying LaTeX files are retained as manuscript-supporting project materials and may predate final v3 wording.

## Data provenance and responsible use

The source team and player box scores are publicly accessible through Euroleague Basketball’s statistics service, with game and play-by-play records available through its Game Center. The repository stores analysis-ready extracts rather than the complete raw response cache.

The processed team table reconstructs all 4,036 official team PIR values exactly. No target-season observation enters its own selection, fitting, or RMS-matching window.

The underlying records are not owned by the authors. Anyone redistributing the data or refreshing the source cache should review the provider’s current terms of use. The collector requires no authentication, and the repository contains no credentials or API secrets.

## Interpretation and limitations

RM-PIR is an alternative allocation of credit within a transparent recorded-action ledger. It is not:

- a causal estimate of player contribution;
- an adjusted plus-minus or lineup-impact model;
- a replacement for video, tactical, tracking, or spatial analysis;
- proof of predictive superiority; or
- prospective validation.

The model class and analytical choices were developed after parts of the historical period had been inspected. A genuinely prospective test requires freezing the complete specification before evaluating an untouched future season.

## Authors and declarations

- **Panagiotis Tirchas:** simulations, software and code development, data analysis, validation, graph and figure generation, and manuscript drafting and review.
- **Nicholas Christakis:** conceptualisation, theory development, methodology, validation, analysis, and manuscript drafting and review.
- **Dimitris Drikakis:** conceptualisation, formal analysis, and manuscript review.

All authors are affiliated with the **Institute for Advanced Modelling and Simulation, University of Nicosia, Nicosia, CY-2417, Cyprus**.

The study reports no specific funding and no competing interests.

## Citation

Until a journal citation and DOI are assigned, cite the manuscript as:

> Tirchas, P., Christakis, N. & Drikakis, D. *Recalibrating EuroLeague Performance Index Rating for outcome-concordant team ledgers: a season-locked historical evaluation*. Manuscript (2026).

## License

This project is released under the GNU General Public License v3.0 or later. See `LICENSE` for the full license text.
