"""Collect official EuroLeague game reports, box scores, and play-by-play.

The collector uses public Euroleague Basketball endpoints, caches every valid
response as gzip-compressed JSON, and records a machine-readable provenance
manifest. Invalid game codes return HTTP 404 and are not cached.
"""

from __future__ import annotations

import argparse
import gzip
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import requests


V3_BASE = "https://api-live.euroleague.net/v3"
PBP_BASE = "https://live.euroleague.net/api/PlaybyPlay"
USER_AGENT = "EuroLeaguePIRTimeAdjustmentResearch/1.0 (non-commercial academic analysis)"
HEADERS = {"User-Agent": USER_AGENT, "Accept": "application/json"}
ROOT = Path(__file__).resolve().parent.parent


def cache_path(root: Path, season: str, kind: str, game_code: int) -> Path:
    return root / season / kind / f"{game_code:04d}.json.gz"


def write_json_gz(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))


def read_json_gz(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def request_json(url: str, params: Optional[Dict[str, Any]] = None, retries: int = 4) -> Tuple[int, Any]:
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, headers=HEADERS, timeout=45)
            if response.status_code == 404:
                return 404, None
            if response.status_code == 429 or response.status_code >= 500:
                if attempt + 1 < retries:
                    time.sleep(0.5 * (2 ** attempt))
                    continue
            response.raise_for_status()
            return response.status_code, response.json()
        except (requests.RequestException, ValueError):
            if attempt + 1 == retries:
                raise
            time.sleep(0.5 * (2 ** attempt))
    raise RuntimeError("unreachable retry state")


def get_or_fetch(path: Path, url: str, params: Optional[Dict[str, Any]], force: bool) -> Tuple[int, Any, bool]:
    if path.exists() and not force:
        return 200, read_json_gz(path), True
    status, payload = request_json(url, params=params)
    if status == 200:
        write_json_gz(path, payload)
    return status, payload, False


def collect_season(
    season: str,
    cache_root: Path,
    max_game_code: int,
    workers: int,
    include_pbp: bool,
    force: bool,
) -> Dict[str, Any]:
    report_url = f"{V3_BASE}/competitions/E/seasons/{season}/games/{{game_code}}/report"
    stats_url = f"{V3_BASE}/competitions/E/seasons/{season}/games/{{game_code}}/stats"

    def report_task(game_code: int) -> Tuple[int, int, Any, bool]:
        path = cache_path(cache_root, season, "report", game_code)
        status, payload, cached = get_or_fetch(
            path, report_url.format(game_code=game_code), None, force
        )
        return game_code, status, payload, cached

    reports: Dict[int, Any] = {}
    cached_reports = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(report_task, code) for code in range(1, max_game_code + 1)]
        for completed, future in enumerate(as_completed(futures), 1):
            code, status, payload, cached = future.result()
            if status == 200 and isinstance(payload, dict) and payload.get("played"):
                reports[code] = payload
            cached_reports += int(cached)
            if completed % 100 == 0:
                print(f"{season}: checked {completed}/{max_game_code} report codes")

    def game_task(item: Tuple[int, Any]) -> Tuple[int, bool, bool, bool, bool]:
        game_code, _report = item
        stats_path = cache_path(cache_root, season, "stats", game_code)
        stats_status, _, stats_cached = get_or_fetch(
            stats_path, stats_url.format(game_code=game_code), None, force
        )
        pbp_ok = False
        pbp_cached = False
        if include_pbp:
            pbp_path = cache_path(cache_root, season, "pbp", game_code)
            pbp_status, pbp_payload, pbp_cached = get_or_fetch(
                pbp_path,
                PBP_BASE,
                {"gamecode": game_code, "seasoncode": season},
                force,
            )
            pbp_ok = pbp_status == 200 and isinstance(pbp_payload, dict)
        return game_code, stats_status == 200, stats_cached, pbp_ok, pbp_cached

    stats_ok = stats_cached_count = pbp_ok_count = pbp_cached_count = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(game_task, item) for item in sorted(reports.items())]
        for completed, future in enumerate(as_completed(futures), 1):
            _, ok, cached, pok, pcached = future.result()
            stats_ok += int(ok)
            stats_cached_count += int(cached)
            pbp_ok_count += int(pok)
            pbp_cached_count += int(pcached)
            if completed % 100 == 0:
                print(f"{season}: collected {completed}/{len(reports)} played games")

    return {
        "season": season,
        "played_reports": len(reports),
        "stats_responses": stats_ok,
        "pbp_responses": pbp_ok_count if include_pbp else 0,
        "cached_reports": cached_reports,
        "cached_stats": stats_cached_count,
        "cached_pbp": pbp_cached_count if include_pbp else 0,
        "max_game_code_checked": max_game_code,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seasons",
        nargs="+",
        default=["E2020", "E2021", "E2022", "E2023", "E2024", "E2025"],
    )
    parser.add_argument("--pbp-seasons", nargs="*", default=["E2025"])
    parser.add_argument("--cache-root", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--max-game-code", type=int, default=500)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    started = datetime.now(timezone.utc)
    summaries = []
    for season in args.seasons:
        summaries.append(
            collect_season(
                season=season,
                cache_root=args.cache_root,
                max_game_code=args.max_game_code,
                workers=max(1, args.workers),
                include_pbp=season in set(args.pbp_seasons),
                force=args.force,
            )
        )

    manifest = {
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "started_at_utc": started.isoformat(timespec="seconds"),
        "competition": "EuroLeague (E)",
        "user_agent": USER_AGENT,
        "sources": {
            "game_report": f"{V3_BASE}/competitions/E/seasons/{{season}}/games/{{game_code}}/report",
            "game_stats": f"{V3_BASE}/competitions/E/seasons/{{season}}/games/{{game_code}}/stats",
            "play_by_play": f"{PBP_BASE}?gamecode={{game_code}}&seasoncode={{season}}",
        },
        "seasons": summaries,
    }
    manifest_path = args.cache_root.parent / "collection_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
