"""Core definitions for time- and possession-adjusted EuroLeague PIR.

The functions in this module are deliberately independent of pandas so that
the formula implementation can be unit-tested against raw API responses.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple


OFFENSE_EVENT_TYPES = {
    "2FGA", "2FGM", "3FGA", "3FGM", "FTA", "FTM", "O", "AS", "TO", "OF", "RV", "AG"
}
# ``AG`` is attached to the offensive player whose shot was rejected;
# ``FV`` is the corresponding defensive block event.
DEFENSE_EVENT_TYPES = {"D", "ST", "FV"}
PERIOD_END_TYPES = {"EP", "EG"}


def minutes_to_decimal(value: Any) -> float:
    """Convert minutes represented as seconds, ``MM:SS``, or a number.

    Raw game reports use seconds and therefore should normally be divided by
    60 directly. This helper is for public table representations such as
    ``23:30``.
    """

    if value is None or value == "":
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if ":" in text:
        minutes, seconds = text.split(":", 1)
        return float(minutes) + float(seconds) / 60.0
    return float(text)


def official_pir(stats: Mapping[str, Any]) -> float:
    """Reconstruct official PIR from one game-level player box score."""

    points = float(stats.get("points", 0) or 0)
    rebounds = float(stats.get("totalRebounds", 0) or 0)
    assists = float(stats.get("assistances", 0) or 0)
    steals = float(stats.get("steals", 0) or 0)
    blocks = float(stats.get("blocksFavour", 0) or 0)
    fouls_drawn = float(stats.get("foulsReceived", 0) or 0)

    fgm = float(stats.get("fieldGoalsMadeTotal", 0) or 0)
    fga = float(stats.get("fieldGoalsAttemptedTotal", 0) or 0)
    ftm = float(stats.get("freeThrowsMade", 0) or 0)
    fta = float(stats.get("freeThrowsAttempted", 0) or 0)
    turnovers = float(stats.get("turnovers", 0) or 0)
    shots_rejected = float(stats.get("blocksAgainst", 0) or 0)
    fouls_committed = float(stats.get("foulsCommited", 0) or 0)

    positive = points + rebounds + assists + steals + blocks + fouls_drawn
    negative = (fga - fgm) + (fta - ftm) + turnovers + shots_rejected + fouls_committed
    return positive - negative


def one_team_box_possessions(stats: Mapping[str, Any], ft_weight: float = 0.44) -> float:
    """Return the unbalanced box-score possession estimate for one team."""

    fga = float(stats.get("fieldGoalsAttemptedTotal", 0) or 0)
    fta = float(stats.get("freeThrowsAttempted", 0) or 0)
    oreb = float(stats.get("offensiveRebounds", 0) or 0)
    turnovers = float(stats.get("turnovers", 0) or 0)
    return fga + ft_weight * fta - oreb + turnovers


def team_possessions(
    team_stats: Mapping[str, Any],
    opponent_stats: Mapping[str, Any],
    ft_weight: float = 0.44,
) -> float:
    """Balanced team possessions used in the recommendation document."""

    return 0.5 * (
        one_team_box_possessions(team_stats, ft_weight)
        + one_team_box_possessions(opponent_stats, ft_weight)
    )


def player_possessions(team_poss: float, minutes: float, game_minutes: float = 40.0) -> float:
    """Allocate team possessions in proportion to a player's court time."""

    if game_minutes <= 0:
        return 0.0
    return float(team_poss) * float(minutes) / float(game_minutes)


def pir_per_40(pir: float, minutes: float) -> float:
    return 40.0 * float(pir) / float(minutes) if minutes > 0 else float("nan")


def pir_per_100(pir: float, possessions: float) -> float:
    return 100.0 * float(pir) / float(possessions) if possessions > 0 else float("nan")


def adjusted_pir100(pir100: float, possessions: float, role_mean: float, k: float) -> float:
    """Reliability-adjusted PIR per 100 possessions."""

    if possessions < 0 or k < 0:
        raise ValueError("possessions and k must be non-negative")
    denominator = possessions + k
    weight = possessions / denominator if denominator > 0 else 1.0
    return float(role_mean) + weight * (float(pir100) - float(role_mean))


def pir_above_average(pir: float, possessions: float, role_mean: float) -> float:
    return float(pir) - float(role_mean) * float(possessions) / 100.0


def _team_code_from_side(side: Mapping[str, Any]) -> str:
    for row in side.get("players", []):
        code = (
            row.get("player", {})
            .get("club", {})
            .get("code")
        )
        if code:
            return str(code).strip()
    return ""


def _player_code(row: Mapping[str, Any]) -> str:
    return str(
        row.get("player", {}).get("person", {}).get("code", "")
    ).strip()


def flatten_pbp_events(payload: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    """Return play-by-play events in their supplied chronological order."""

    out: List[Mapping[str, Any]] = []

    def append_events(value: Any) -> None:
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict) and "PLAYTYPE" in item:
                    out.append(item)
                elif isinstance(item, list):
                    append_events(item)

    for key in ("FirstQuarter", "SecondQuarter", "ThirdQuarter", "ForthQuarter", "ExtraTime"):
        append_events(payload.get(key, []))
    return out


def reconstruct_on_court_possessions(
    pbp: Mapping[str, Any], game_stats: Mapping[str, Any]
) -> Dict[str, Any]:
    """Count offensive possessions experienced by each on-court player.

    The state machine identifies possession changes from the event stream and
    applies each completed possession to the offensive lineup at its last
    event. Initial lineups come from the official ``startFive`` flags and are
    updated with official ``IN``/``OUT`` events. The routine also returns
    lineup-integrity diagnostics; observations from malformed lineups remain
    visible rather than being silently treated as exact.
    """

    sides = ("local", "road")
    team_codes = [_team_code_from_side(game_stats.get(side, {})) for side in sides]
    team_codes = [code for code in team_codes if code]
    if len(team_codes) != 2:
        raise ValueError("could not identify both team codes from game stats")

    lineups: Dict[str, set] = {code: set() for code in team_codes}
    for side, team_code in zip(sides, team_codes):
        for row in game_stats.get(side, {}).get("players", []):
            if bool(row.get("stats", {}).get("startFive")):
                code = _player_code(row)
                if code:
                    lineups[team_code].add(code)

    player_counts: MutableMapping[Tuple[str, str], int] = defaultdict(int)
    team_counts: MutableMapping[str, int] = defaultdict(int)
    bad_lineup_possessions = 0
    lineup_sizes: MutableMapping[str, List[int]] = defaultdict(list)
    possession_records: List[Dict[str, Any]] = []

    current_offense: str | None = None
    last_lineup: Tuple[str, ...] = ()
    last_event_number: Any = None

    def opponent(code: str) -> str:
        return team_codes[1] if code == team_codes[0] else team_codes[0]

    def close_possession(reason: str) -> None:
        nonlocal bad_lineup_possessions
        if not current_offense:
            return
        lineup = tuple(last_lineup)
        lineup_sizes[current_offense].append(len(lineup))
        if len(lineup) != 5:
            bad_lineup_possessions += 1
        team_counts[current_offense] += 1
        for code in lineup:
            player_counts[(current_offense, code)] += 1
        possession_records.append(
            {
                "team_code": current_offense,
                "players": lineup,
                "end_event": last_event_number,
                "reason": reason,
                "lineup_size": len(lineup),
            }
        )

    for event in flatten_pbp_events(pbp):
        play_type = str(event.get("PLAYTYPE", "")).strip().upper()
        team_code = str(event.get("CODETEAM", "")).strip()
        player_id = str(event.get("PLAYER_ID", "")).strip()
        if player_id.startswith("P"):
            player_id = player_id[1:]

        if play_type == "OUT" and team_code in lineups and player_id:
            lineups[team_code].discard(player_id)
            continue
        if play_type == "IN" and team_code in lineups and player_id:
            lineups[team_code].add(player_id)
            continue
        if play_type in PERIOD_END_TYPES:
            close_possession("period_end")
            current_offense = None
            last_lineup = ()
            last_event_number = None
            continue

        candidate: str | None = None
        if team_code in team_codes and play_type in OFFENSE_EVENT_TYPES:
            candidate = team_code
        elif team_code in team_codes and play_type in DEFENSE_EVENT_TYPES:
            candidate = opponent(team_code)
        if candidate is None:
            continue

        if current_offense is not None and candidate != current_offense:
            close_possession("team_change")
        current_offense = candidate
        last_lineup = tuple(sorted(lineups[candidate]))
        last_event_number = event.get("NUMBEROFPLAY")

    close_possession("feed_end")
    return {
        "player_possessions": dict(player_counts),
        "team_possessions": dict(team_counts),
        "bad_lineup_possessions": bad_lineup_possessions,
        "lineup_sizes": dict(lineup_sizes),
        "records": possession_records,
    }
