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
    inspect = commands.add_parser("inspect")
    inspect.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    inspect.add_argument("--power", type=Path)
    inspect.add_argument("--weather", type=Path)
    inspect.add_argument("--target", choices=TARGETS, default="cooling_meter")
    inspect.add_argument("--audit", type=Path, default=Path("data/processed/audit.json"))
    args = parser.parse_args()
    try:
        if args.command == "download":
            download(args.data_dir)
        else:
            power = input_path(args.data_dir, "power", args.power)
            weather = input_path(args.data_dir, "weather", args.weather)
            _, report = prepare(power, weather, args.target, 0.75)
            write_json(args.audit, report)
            print(args.audit)
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
