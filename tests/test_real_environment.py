from pathlib import Path
import tomllib
from unittest import TestCase

from graphiti_text_adventure.real_environment import (
    DEFAULT_REAL_ENV,
    REQUIRED_REAL_ENV,
    load_project_dotenv,
    validate_real_environment,
)


class RealEnvironmentTests(TestCase):
    def test_validate_real_environment_reports_missing_required_values(self):
        report = validate_real_environment({"OPENAI_API_KEY": "test-key"})

        self.assertEqual(report.missing, ["NEO4J_PASSWORD"])
        self.assertEqual(report.values["NEO4J_URI"], "bolt://localhost:7687")
        self.assertEqual(report.values["NEO4J_USER"], "neo4j")
        self.assertEqual(report.values["OPENAI_BASE_URL"], "https://api.openai.com/v1")

    def test_env_example_documents_required_and_defaulted_values(self):
        env_example = Path(".env.example").read_text(encoding="utf-8")

        for key in [*REQUIRED_REAL_ENV, *DEFAULT_REAL_ENV]:
            self.assertIn(f"{key}=", env_example)

    def test_load_project_dotenv_accepts_missing_file(self):
        self.assertFalse(load_project_dotenv(Path("missing.env")))

    def test_readme_uses_existing_real_services_without_docker(self):
        readme = Path("README.md").read_text(encoding="utf-8")

        self.assertIn("确保 Neo4j 已经在本机或远端启动", readme)
        self.assertNotIn("docker compose", readme)
        self.assertFalse(Path("docker-compose.yml").exists())

    def test_graphiti_runtime_dependencies_are_optional(self):
        pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

        dependencies = pyproject["project"].get("dependencies", [])
        real_dependencies = pyproject["project"]["optional-dependencies"]["real"]
        self.assertNotIn("graphiti-core>=0.29.0", dependencies)
        self.assertIn("graphiti-core>=0.29.0", real_dependencies)
        self.assertIn("neo4j>=6.1.0", real_dependencies)
