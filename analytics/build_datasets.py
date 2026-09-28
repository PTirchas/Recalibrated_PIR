"""Transform cached official responses into analysis-ready game tables."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import pandas as pd

from .metrics import (
    official_pir,
    player_possessions,
    reconstruct_on_court_possessions,
    team_possessions,
)


ROOT = Path(__file__).resolve().parent.parent
TEAM_SOURCE_FIELDS = {
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


def read_json(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def team_code(side_payload: Mapping[str, Any]) -> str:
    for row in side_payload.get("players", []):
        code = row.get("player", {}).get("club", {}).get("code")
        if code:
            return str(code).strip()
    return ""


def game_rows(
    season: str,
    game_code: int,
    report: Mapping[str, Any],
    stats: Mapping[str, Any],
    pbp: Mapping[str, Any] | None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any] | None]:
    local = stats.get("local", {})
    road = stats.get("road", {})
    side_payloads = {"local": local, "road": road}
    codes = {side: team_code(payload) for side, payload in side_payloads.items()}
    if not codes["local"] or not codes["road"]:
        raise ValueError(f"{season}/{game_code}: missing team code")

    local_total = local.get("total", {})
    road_total = road.get("total", {})
    estimated_poss = team_possessions(local_total, road_total)
    duration = max(
        float(local_total.get("timePlayed", 2400) or 2400),
        float(road_total.get("timePlayed", 2400) or 2400),
    ) / 60.0

    pbp_result = reconstruct_on_court_possessions(pbp, stats) if pbp else None
    player_pbp = pbp_result["player_possessions"] if pbp_result else {}
    team_pbp = pbp_result["team_possessions"] if pbp_result else {}

    common = {
        "Season": season,
        "GameCode": game_code,
        "Date": report.get("utcDate") or report.get("date"),
        "Round": report.get("round"),
        "Phase": report.get("phaseType", {}).get("name"),
        "GameMinutes": duration,
        "EstimatedTeamPoss": estimated_poss,
    }
    team_rows: List[Dict[str, Any]] = []
    player_rows: List[Dict[str, Any]] = []

    for side in ("local", "road"):
        opponent_side = "road" if side == "local" else "local"
        payload = side_payloads[side]
        total = payload.get("total", {})
        code = codes[side]
        opp_code = codes[opponent_side]
        club = report.get(side, {}).get("club", {})
        team_rows.append(
            {
                **common,
                "Side": side,
                "TeamCode": code,
                "OpponentCode": opp_code,
                "TeamName": club.get("abbreviatedName") or club.get("name") or code,
                **{name: total.get(field) for name, field in TEAM_SOURCE_FIELDS.items()},
                "PBPTeamPoss": team_pbp.get(code) if pbp_result else None,
                "PBPBadLineupPoss": pbp_result.get("bad_lineup_possessions") if pbp_result else None,
            }
        )
        for row in payload.get("players", []):
            pstats = row.get("stats", {})
            seconds = float(pstats.get("timePlayed", 0) or 0)
            if seconds <= 0:
                continue
            pobj = row.get("player", {})
            person = pobj.get("person", {})
            pcode = str(person.get("code", "")).strip()
            minutes = seconds / 60.0
            official_value = float(pstats.get("valuation", 0) or 0)
            reconstructed_value = official_pir(pstats)
            player_rows.append(
                {
                    **common,
                    "Side": side,
                    "TeamCode": code,
                    "OpponentCode": opp_code,
                    "PlayerCode": pcode,
                    "Player": person.get("name"),
                    "Role": pobj.get("positionName") or "Unknown",
                    "PositionCode": pobj.get("position"),
                    "Starter": bool(pstats.get("startFive")),
                    "Minutes": minutes,
                    "PIR": official_value,
                    "PlusMinus": pstats.get("plusMinus"),
                    "PIRReconstructed": reconstructed_value,
                    "PIRDifference": official_value - reconstructed_value,
                    "PlayerPossDoc40": player_possessions(estimated_poss, minutes, 40.0),
                    "PlayerPossEstimated": player_possessions(estimated_poss, minutes, duration),
                    "PlayerPossPBP": player_pbp.get((code, pcode), 0) if pbp_result else None,
                    "PTS": pstats.get("points"),
                    "TR": pstats.get("totalRebounds"),
                    "AST": pstats.get("assistances"),
                    "STL": pstats.get("steals"),
                    "BLK": pstats.get("blocksFavour"),
                    "FD": pstats.get("foulsReceived"),
                    "FGA": pstats.get("fieldGoalsAttemptedTotal"),
                    "FGM": pstats.get("fieldGoalsMadeTotal"),
                    "FTA": pstats.get("freeThrowsAttempted"),
                    "FTM": pstats.get("freeThrowsMade"),
                    "TO": pstats.get("turnovers"),
                    "BLKA": pstats.get("blocksAgainst"),
                    "FC": pstats.get("foulsCommited"),
                }
            )
    return player_rows, team_rows, pbp_result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument(
        "--seasons",
        nargs="+",
        default=["E2020", "E2021", "E2022", "E2023", "E2024", "E2025"],
    )
    args = parser.parse_args()

    players: List[Dict[str, Any]] = []
    teams: List[Dict[str, Any]] = []
    qc: List[Dict[str, Any]] = []
    exclusions: List[Dict[str, Any]] = []
    for season in args.seasons:
        report_dir = args.cache_root / season / "report"
        for report_path in sorted(report_dir.glob("*.json.gz")):
            report = read_json(report_path)
            if report.get("played", False):
                continue
            local = report.get("local", {}).get("club", {}).get("code", "")
            road = report.get("road", {}).get("club", {}).get("code", "")
            exclusions.append(
                {
                    "Season": season,
                    "GameCode": int(report.get("gameCode", report_path.name.split(".", 1)[0])),
                    "Date": report.get("date"),
                    "Teams": f"{local}-{road}",
                    "Scope": "Primary team analysis",
                    "Reason": "Official report played=false and no completed-game statistics; payload contains no cancellation-reason field",
                }
            )
        stats_dir = args.cache_root / season / "stats"
        for stats_path in sorted(stats_dir.glob("*.json.gz")):
            game_code = int(stats_path.name.split(".", 1)[0])
            report_path = args.cache_root / season / "report" / stats_path.name
            if not report_path.exists():
                continue
            pbp_path = args.cache_root / season / "pbp" / stats_path.name
            report = read_json(report_path)
            stats = read_json(stats_path)
            pbp = read_json(pbp_path) if pbp_path.exists() else None
            try:
                prows, trows, pbp_result = game_rows(
                    season, game_code, report, stats, pbp
                )
            except (KeyError, TypeError, ValueError) as exc:
                qc.append({"Season": season, "GameCode": game_code, "Status": "error", "Detail": str(exc)})
                continue
            players.extend(prows)
            teams.extend(trows)
            bad_lineups = pbp_result.get("bad_lineup_possessions", 0) if pbp_result else 0
            if bad_lineups:
                exclusions.append(
                    {
                        "Season": season,
                        "GameCode": game_code,
                        "Date": report.get("date"),
                        "Teams": f"{trows[0]['TeamCode']}-{trows[1]['TeamCode']}",
                        "Scope": "Secondary play-by-play possession-fit comparison only",
                        "Reason": "Malformed-lineup possession(s); retained in all box-score and RM-PIR analyses",
                    }
                )
            qc.append(
                {
                    "Season": season,
                    "GameCode": game_code,
                    "Status": "ok",
                    "Players": len(prows),
                    "HasPBP": pbp_result is not None,
                    "PBPBadLineupPoss": pbp_result.get("bad_lineup_possessions") if pbp_result else None,
                }
            )
        print(f"{season}: accumulated {sum(r['Season'] == season for r in players)} player-games")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    player_df = pd.DataFrame(players).sort_values(["Season", "Date", "GameCode", "TeamCode", "PlayerCode"])
    team_df = pd.DataFrame(teams).sort_values(["Season", "Date", "GameCode", "TeamCode"])
    qc_df = pd.DataFrame(qc).sort_values(["Season", "GameCode"])
    name_df = (
        team_df[["TeamCode", "TeamName"]]
        .drop_duplicates("TeamCode", keep="last")
        .sort_values("TeamCode")
    )
    exclusion_df = pd.DataFrame(exclusions).sort_values(["Season", "GameCode"])
    player_df.to_csv(args.output_dir / "player_games.csv", index=False)
    team_df.to_csv(args.output_dir / "team_boxscores.csv", index=False)
    name_df.to_csv(args.output_dir / "team_names.csv", index=False)
    exclusion_df.to_csv(args.output_dir / "source_exclusions.csv", index=False)
    qc_df.to_csv(args.output_dir / "build_qc.csv", index=False)
    print(f"wrote {len(player_df):,} player-games and {len(team_df):,} team-games")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
