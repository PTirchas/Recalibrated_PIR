"""Run the complete RM-PIR reproduction workflow."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SEASONS = ("E2020", "E2021", "E2022", "E2023", "E2024", "E2025")


def run(arguments: list[str]) -> None:
    print("\n>>>", " ".join(arguments), flush=True)
    subprocess.run(arguments, cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="download the official-source cache and rebuild processed inputs",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="check the bundled outputs without rerunning the analyses",
    )
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--max-game-code", type=int, default=500)
    args = parser.parse_args()

    python = sys.executable
    if args.verify_only:
        run([python, "-m", "unittest", "discover", "-s", "tests", "-v"])
        return 0

    if args.refresh_data:
        run(
            [
                python,
                "-m",
                "analytics.collect_data",
                "--seasons",
                *SEASONS,
                "--pbp-seasons",
                "E2025",
                "--workers",
                str(max(1, args.workers)),
                "--max-game-code",
                str(args.max_game_code),
            ]
        )
        run([python, "-m", "analytics.build_datasets", "--seasons", *SEASONS])

    run([python, "-m", "analytics.rm_pir_analysis"])
    run([python, "-m", "analytics.technical_robustness"])
    run([python, "-m", "unittest", "discover", "-s", "tests", "-v"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

