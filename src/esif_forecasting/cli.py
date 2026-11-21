import argparse
import json
from pathlib import Path

from .data import TARGETS, input_path, prepare
from .download import download


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="ESIF auxiliary-power forecasting")
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("download")
    fetch.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    for command in ("inspect", "baseline"):
        subparser = commands.add_parser(command)
        subparser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
        subparser.add_argument("--power", type=Path)
        subparser.add_argument("--weather", type=Path)
        subparser.add_argument("--target", choices=TARGETS, default="cooling_meter")
    commands.choices["inspect"].add_argument(
        "--audit", type=Path, default=Path("data/processed/audit.json"))
    commands.choices["baseline"].add_argument(
        "--output", type=Path, default=Path("results/baseline_predictions.csv"))
    experiment = commands.add_parser("run")
    experiment.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    experiment.add_argument("--power", type=Path)
    experiment.add_argument("--weather", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "download":
            download(args.data_dir)
        elif args.command == "run":
            from .experiment import run

            run(args.config, args.power, args.weather)
        else:
            power = input_path(args.data_dir, "power", args.power)
            weather = input_path(args.data_dir, "weather", args.weather)
            hourly, report = prepare(power, weather, args.target, 0.75)
            if args.command == "inspect":
                write_json(args.audit, report)
                print(args.audit)
            else:
                from .evaluation import baseline_predictions, evaluation_folds

                scheme, folds = evaluation_folds(hourly, 4, 0.95)
                predictions = baseline_predictions(hourly, folds)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                predictions.to_csv(args.output, index=False)
                print(f"{scheme}: {args.output}")
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
