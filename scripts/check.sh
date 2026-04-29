#!/usr/bin/env bash
set -euo pipefail

export PYTHONDONTWRITEBYTECODE=1
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN=python3
fi

"$PYTHON_BIN" -m unittest discover -s tests -v
"$PYTHON_BIN" -c 'import ast, pathlib; [ast.parse(path.read_text(encoding="utf-8"), filename=str(path)) for path in pathlib.Path("graphiti_text_adventure").glob("*.py")]'
"$PYTHON_BIN" -m graphiti_text_adventure.cli validate-world examples/stonebridge_world.json >/dev/null
