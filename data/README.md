# Data

`processed/` contains the frozen inputs used by the reported analysis:

- `team_boxscores.csv`: 4,036 official team-game rows with the complete PIR action ledger and game metadata.
- `player_games.csv`: 43,549 positive-minute player-game rows used for player decomposition and secondary diagnostics.
- `team_names.csv`: stable team-code display-name mapping.
- `source_exclusions.csv`: cancelled-report and play-by-play-quality exclusions used by the audit.

Season codes map as follows: E2020 = 2020–21, E2021 = 2021–22, E2022 = 2022–23, E2023 = 2023–24, E2024 = 2024–25, and E2025 = 2025–26.

The public source endpoints are recorded by the collector:

- game reports: `https://api-live.euroleague.net/v3/competitions/E/seasons/{season}/games/{game_code}/report`
- box scores: `https://api-live.euroleague.net/v3/competitions/E/seasons/{season}/games/{game_code}/stats`
- play-by-play: `https://live.euroleague.net/api/PlaybyPlay?gamecode={game_code}&seasoncode={season}`

The original collections were retrieved on 1 July 2026 (E2023–E2025) and 22 July 2026 (E2020–E2022). Their collection manifests are retained under `provenance/`. SHA-256 hashes of the exact processed inputs used by the analysis are in `../tables/analysis_manifest.json`.

Fresh API responses are written to `raw/`, which is intentionally git-ignored. Rebuild the processed tables with:

```bash
python -m analytics.collect_data
python -m analytics.build_datasets
```

