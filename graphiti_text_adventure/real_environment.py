from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


REQUIRED_REAL_ENV = ("NEO4J_PASSWORD", "OPENAI_API_KEY")
DEFAULT_REAL_ENV = {
    "NEO4J_URI": "bolt://localhost:7687",
    "NEO4J_USER": "neo4j",
    "OPENAI_BASE_URL": "https://api.openai.com/v1",
    "OPENAI_MODEL": "gpt-4.1-mini",
    "GRAPHITI_SMOKE_GROUP": "graphiti_text_adventure_smoke",
}


@dataclass(frozen=True)
class RealEnvironmentReport:
    values: dict[str, str]
    missing: list[str]

    @property
    def ok(self) -> bool:
        return not self.missing


def load_project_dotenv(path: Path | str = ".env") -> bool:
    env_path = Path(path)
    if not env_path.exists():
        return False
    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError as error:
        raise RuntimeError("python-dotenv is required to load .env files") from error
    load_dotenv(env_path)
    return True


def validate_real_environment(environ: Mapping[str, str] | None = None) -> RealEnvironmentReport:
    source = os.environ if environ is None else environ
    values = {key: source.get(key, default) for key, default in DEFAULT_REAL_ENV.items()}
    for key in REQUIRED_REAL_ENV:
        values[key] = source.get(key, "")
    missing = [key for key in REQUIRED_REAL_ENV if not values[key]]
    return RealEnvironmentReport(values=values, missing=missing)


def require_real_environment(environ: Mapping[str, str] | None = None) -> RealEnvironmentReport:
    report = validate_real_environment(environ)
    if not report.ok:
        missing = ", ".join(report.missing)
        raise RuntimeError(f"Missing required real environment variables: {missing}")
    return report
