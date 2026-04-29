import asyncio
import importlib.util
import os
import unittest
from unittest import TestCase
from unittest.mock import patch

from graphiti_text_adventure.runtime_clients import (
    CompatibleOpenAIGenericClient,
    HashEmbedder,
    SimpleReranker,
    build_runtime_clients_from_env,
)


class RuntimeClientTests(TestCase):
    def test_repair_response_normalizes_common_entity_schema_drift(self):
        data = {
            "entities": [
                {"entity_name": "张守义", "entity_type": "Entity"},
                {"entity_name": "何桂兰", "entity_type_name": "Entity"},
            ]
        }

        repaired = CompatibleOpenAIGenericClient.repair_response(
            data,
            expected_fields={"extracted_entities"},
        )

        self.assertEqual(
            repaired,
            {
                "extracted_entities": [
                    {"name": "张守义", "entity_type_id": 0},
                    {"name": "何桂兰", "entity_type_id": 0},
                ]
            },
        )

    def test_repair_response_normalizes_common_edge_schema_drift(self):
        data = {
            "relationships": [
                {
                    "from": "张守义",
                    "to": "何桂兰",
                    "type": "DISTRUSTS",
                    "fact": "张守义不信任何桂兰。",
                }
            ]
        }

        repaired = CompatibleOpenAIGenericClient.repair_response(data, expected_fields={"edges"})

        self.assertEqual(
            repaired["edges"][0],
            {
                "source_entity_name": "张守义",
                "target_entity_name": "何桂兰",
                "relation_type": "DISTRUSTS",
                "fact": "张守义不信任何桂兰。",
            },
        )

    def test_hash_embedder_is_deterministic_and_uses_requested_dimension(self):
        embedder = HashEmbedder(dim=8)

        first = asyncio.run(embedder.create("张守义"))
        second = asyncio.run(embedder.create("张守义"))
        other = asyncio.run(embedder.create("何桂兰"))

        self.assertEqual(first, second)
        self.assertNotEqual(first, other)
        self.assertEqual(len(first), 8)

    def test_simple_reranker_returns_all_passages_in_original_order(self):
        reranker = SimpleReranker()

        ranked = asyncio.run(reranker.rank("query", ["a", "b", "c"]))

        self.assertEqual(ranked, [("a", 1.0), ("b", 0.5), ("c", 1 / 3)])

    @unittest.skipUnless(importlib.util.find_spec("graphiti_core"), "graphiti-core is optional")
    def test_runtime_clients_default_to_public_openai_base_url(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
            llm, _embedder, _reranker = build_runtime_clients_from_env()

        self.assertEqual(llm.config.base_url, "https://api.openai.com/v1")
