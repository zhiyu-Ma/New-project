from __future__ import annotations

import argparse
import json
from pathlib import Path

from .prompts import INITIALIZATION_PROMPT
from .world import episodes_from_world, validate_world_data, world_from_dict


def main() -> None:
    parser = argparse.ArgumentParser(prog="graphiti-text-adventure")
    subcommands = parser.add_subparsers(dest="command", required=True)

    validate_parser = subcommands.add_parser("validate-world", help="validate generated world JSON")
    validate_parser.add_argument("path", type=Path)

    episodes_parser = subcommands.add_parser("episodes", help="print initialization episodes as JSONL")
    episodes_parser.add_argument("path", type=Path)

    subcommands.add_parser("init-prompt", help="print the world initialization prompt")

    args = parser.parse_args()
    if args.command == "init-prompt":
        print(INITIALIZATION_PROMPT)
    elif args.command == "validate-world":
        data = _load_json(args.path)
        report = validate_world_data(data)
        print(json.dumps(report.__dict__, ensure_ascii=False, indent=2))
    elif args.command == "episodes":
        world = world_from_dict(_load_json(args.path))
        for episode in episodes_from_world(world):
            print(
                json.dumps(
                    {
                        "name": episode.name,
                        "body": episode.body,
                        "source": episode.source.value,
                        "source_description": episode.source_description,
                    },
                    ensure_ascii=False,
                )
            )


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


if __name__ == "__main__":
    main()
