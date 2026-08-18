"""Run the 1QB -> Superflex validation workflow."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.analysis import run_superflex_validation
from src.config import OUTPUTS_DIR
from src.sleeper import SleeperClient


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate league-specific ADP model against known Superflex ADP.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_DIR, help="Directory for validation CSV outputs.")
    args = parser.parse_args()

    validation = run_superflex_validation(client=SleeperClient())
    args.output_dir.mkdir(parents=True, exist_ok=True)

    validation["metrics"].to_csv(args.output_dir / "model_metrics.csv", index=False)
    validation["actual_superflex_adp"].to_csv(args.output_dir / "actual_superflex_adp.csv", index=False)
    for model_name, prediction in validation["predictions"].items():
        safe_name = model_name.lower().replace(" ", "_").replace("+", "plus")
        prediction.to_csv(args.output_dir / f"{safe_name}_prediction.csv", index=False)
    for model_name, breakdown in validation["positional_breakdowns"].items():
        safe_name = model_name.lower().replace(" ", "_").replace("+", "plus")
        breakdown.to_csv(args.output_dir / f"{safe_name}_positional_errors.csv", index=False)

    print(validation["metrics"].to_string(index=False))
    print(f"\nSaved outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
