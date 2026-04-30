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

    def test_repair_response_fills_missing_edge_fact(self):
        data = {
            "edges": [
                {
                    "source_entity_name": "张守义",
                    "target_entity_name": "张守义的女儿",
                    "relation_type": "FEARS_FOR",
                }
            ]
        }

        repaired = CompatibleOpenAIGenericClient.repair_response(data, expected_fields={"edges"})

        self.assertEqual(
            repaired["edges"][0],
            {
                "source_entity_name": "张守义",
                "target_entity_name": "张守义的女儿",
                "relation_type": "FEARS_FOR",
                "fact": "张守义 FEARS_FOR 张守义的女儿",
            },
        )

    def test_repair_response_maps_demo_entity_type_names_to_ids(self):
        data = {
            "extracted_entities": [
                {"name": "张守义", "entity_type": "Person"},
                {"name": "镇上", "entity_type": "Location"},
                {"name": "旧账", "entity_type": "Topic"},
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
                    {"name": "张守义", "entity_type_id": 1},
                    {"name": "镇上", "entity_type_id": 2},
                    {"name": "旧账", "entity_type_id": 3},
                ]
            },
        )

    def test_repair_response_accepts_entity_text_as_name(self):
        data = {
            "extracted_entities": [
                {"entity_text": "张守义", "entity_type_id": 1},
                {"entity_text": "张守义的女儿", "entity_type": "Person"},
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
                    {"name": "张守义", "entity_type_id": 1},
                    {"name": "张守义的女儿", "entity_type_id": 1},
                ]
            },
        )

    def test_repair_response_unwraps_and_prunes_attribute_payloads(self):
        data = {
            "name": "张守义",
            "entity_types": ["Entity", "Person"],
            "attributes": {
                "occupation": "铁匠铺老板",
                "current_mood": "焦虑",
            },
        }

        repaired = CompatibleOpenAIGenericClient.repair_response(
            data,
            expected_fields={"personality", "current_mood"},
        )

        self.assertEqual(repaired, {"current_mood": "焦虑"})

    def test_repair_response_drops_nested_attribute_maps(self):
        data = {
            "name": "张守义",
            "entity_types": ["Entity", "Person"],
            "attributes": {
                "current_mood": {"occupation": "铁匠铺老板"},
                "personality": ["固执", "谨慎"],
            },
        }

        repaired = CompatibleOpenAIGenericClient.repair_response(
            data,
            expected_fields={"personality", "current_mood"},
        )

        self.assertEqual(repaired, {"personality": ["固执", "谨慎"]})

    def test_repair_response_drops_custom_property_maps_without_attribute_wrapper(self):
        data = {"strength": {"value": 7}, "note": "ignored"}

        repaired = CompatibleOpenAIGenericClient.repair_response(
            data,
            expected_fields={"strength"},
        )

        self.assertEqual(repaired, {})

    def test_repair_response_splits_personality_string(self):
        data = {"personality": "谨慎、重名声"}

        repaired = CompatibleOpenAIGenericClient.repair_response(
            data,
            expected_fields={"personality", "current_mood"},
        )

        self.assertEqual(repaired, {"personality": ["谨慎", "重名声"]})

    def test_repair_response_defaults_missing_summaries_to_empty_list(self):
        repaired = CompatibleOpenAIGenericClient.repair_response(
            {},
            expected_fields={"summaries"},
        )

        self.assertEqual(repaired, {"summaries": []})

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
