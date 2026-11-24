"""Command line: python -m egrivis run <scenario.json> --seed N"""

import argparse
import sys
from pathlib import Path

from egrivis.run_store import run_and_save
from egrivis.runner import POLICIES
from egrivis.scenario import ScenarioError, load_scenario


def main() -> int:
    parser = argparse.ArgumentParser(prog="egrivis")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run one scenario and save its bundle")
    run.add_argument("scenario", type=Path)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--policy", choices=sorted(POLICIES), default="B1")
    run.add_argument("--out", type=Path, default=Path("runs"))
    args = parser.parse_args()
    try:
        scenario = load_scenario(args.scenario)
    except (ScenarioError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(run_and_save(scenario, args.seed, args.policy, args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
