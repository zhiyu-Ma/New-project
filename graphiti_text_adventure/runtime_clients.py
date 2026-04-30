from __future__ import annotations

import hashlib
import os
import random
from typing import Any

try:
    from graphiti_core.cross_encoder.client import CrossEncoderClient
    from graphiti_core.embedder.client import EmbedderClient
    from graphiti_core.llm_client.config import DEFAULT_MAX_TOKENS, LLMConfig, ModelSize
    from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
    from graphiti_core.prompts.models import Message
    from pydantic import BaseModel
except ModuleNotFoundError:
    CrossEncoderClient = object
    EmbedderClient = object
    OpenAIGenericClient = object
    BaseModel = Any
    Message = Any
    DEFAULT_MAX_TOKENS = 16384

    class ModelSize:
        medium = "medium"

    LLMConfig = None


COMPLEX_RESPONSE_FIELDS = {
    "edges",
    "entity_resolutions",
    "extracted_entities",
    "summaries",
    "timestamps",
}
ENTITY_TYPE_IDS = {
    "Entity": 0,
    "Person": 1,
    "Location": 2,
    "Topic": 3,
    "人物": 1,
    "地点": 2,
    "话题": 3,
}


class CompatibleOpenAIGenericClient(OpenAIGenericClient):
    """OpenAI-compatible chat client with small repairs for weaker schema adherence."""

    def __init__(self, *args: Any, **kwargs: Any):
        if OpenAIGenericClient is object:
            raise RuntimeError("graphiti-core is required to build runtime clients")
        super().__init__(*args, **kwargs)

    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, Any]:
        data = await super()._generate_response(
            messages,
            response_model=response_model,
            max_tokens=max_tokens,
            model_size=model_size,
        )
        expected_fields = set(response_model.model_fields) if response_model is not None else set()
        return self.repair_response(data, expected_fields)

    @staticmethod
    def repair_response(data: dict[str, Any], expected_fields: set[str]) -> dict[str, Any]:
        if "extracted_entities" in expected_fields and "extracted_entities" not in data:
            for key in ("entities", "entity_list"):
                if key in data:
                    data["extracted_entities"] = data.pop(key)
                    break

        if "edges" in expected_fields and "edges" not in data:
            for key in ("relationships", "facts", "extracted_edges"):
                if key in data:
                    data["edges"] = data.pop(key)
                    break

        if "timestamps" in expected_fields and "timestamps" not in data:
            for key in ("edge_timestamps", "time_bounds"):
                if key in data:
                    data["timestamps"] = data.pop(key)
                    break

        if "summaries" in expected_fields and "summaries" not in data:
            for key in ("entity_summaries", "summarized_entities", "summary_list"):
                if key in data:
                    data["summaries"] = data.pop(key)
                    break
            else:
                data["summaries"] = []

        if isinstance(data.get("extracted_entities"), list):
            for item in data["extracted_entities"]:
                if isinstance(item, dict) and "name" not in item:
                    for key in ("entity_name", "entity_text", "entity", "text", "title"):
                        if key in item:
                            item["name"] = item.pop(key)
                            break
                if isinstance(item, dict) and "entity_type_id" not in item:
                    for key in ("entity_type", "entity_type_name", "type"):
                        entity_type = item.get(key)
                        if isinstance(entity_type, str) and entity_type in ENTITY_TYPE_IDS:
                            item["entity_type_id"] = ENTITY_TYPE_IDS[entity_type]
                            break
                if isinstance(item, dict) and "entity_type_id" in item:
                    item.pop("entity_type", None)
                    item.pop("entity_type_name", None)

        if isinstance(data.get("edges"), list):
            for item in data["edges"]:
                if not isinstance(item, dict):
                    continue
                if "source_entity_name" not in item:
                    for key in ("source", "source_entity", "from", "from_entity"):
                        if key in item:
                            item["source_entity_name"] = item.pop(key)
                            break
                if "target_entity_name" not in item:
                    for key in ("target", "target_entity", "to", "to_entity"):
                        if key in item:
                            item["target_entity_name"] = item.pop(key)
                            break
                if "relation_type" not in item:
                    for key in ("relationship", "type", "relation"):
                        if key in item:
                            item["relation_type"] = item.pop(key)
                            break
                if "fact" not in item:
                    for key in ("fact_text", "description", "statement", "relationship_description"):
                        if key in item:
                            item["fact"] = item.pop(key)
                            break
                if "fact" not in item and all(
                    key in item
                    for key in ("source_entity_name", "target_entity_name", "relation_type")
                ):
                    item["fact"] = (
                        f"{item['source_entity_name']} "
                        f"{item['relation_type']} "
                        f"{item['target_entity_name']}"
                    )

        had_attribute_wrapper = isinstance(data.get("attributes"), dict)
        if expected_fields:
            attributes = data.get("attributes")
            if isinstance(attributes, dict):
                for key in expected_fields:
                    if key not in data and key in attributes:
                        data[key] = attributes[key]

            data = {key: value for key, value in data.items() if key in expected_fields}
            if isinstance(data.get("personality"), str):
                data["personality"] = _split_delimited_text(data["personality"])
            if had_attribute_wrapper or expected_fields.isdisjoint(COMPLEX_RESPONSE_FIELDS):
                data = {
                    key: value
                    for key, value in data.items()
                    if _is_neo4j_property_value(value)
                }

        return data


def _is_neo4j_property_value(value: Any) -> bool:
    if isinstance(value, str | int | float | bool):
        return True
    if isinstance(value, list):
        return all(isinstance(item, str | int | float | bool) for item in value)
    return False


def _split_delimited_text(value: str) -> list[str]:
    normalized = value
    for delimiter in ("、", "，", ",", "；", ";"):
        normalized = normalized.replace(delimiter, "|")
    parts = [part.strip() for part in normalized.split("|") if part.strip()]
    return parts or [value]


class HashEmbedder(EmbedderClient):
    """Deterministic local embeddings for Graphiti demos when the LLM proxy has no embeddings API."""

    def __init__(self, dim: int = 1024):
        self.dim = dim

    async def create(self, input_data: Any) -> list[float]:
        text = str(input_data)
        seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
        rng = random.Random(seed)
        vector = [rng.uniform(-1.0, 1.0) for _ in range(self.dim)]
        norm = sum(value * value for value in vector) ** 0.5 or 1.0
        return [value / norm for value in vector]

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        return [await self.create(item) for item in input_data_list]


class SimpleReranker(CrossEncoderClient):
    """Stable no-network reranker; preserves Graphiti search plumbing for smoke tests."""

    async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
        return [(passage, 1.0 / (index + 1)) for index, passage in enumerate(passages)]


def build_runtime_clients_from_env() -> tuple[
    CompatibleOpenAIGenericClient,
    HashEmbedder,
    SimpleReranker,
]:
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")
    model = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY must be set")
    if LLMConfig is None:
        raise RuntimeError("graphiti-core is required to build runtime clients")

    llm = CompatibleOpenAIGenericClient(
        LLMConfig(
            api_key=api_key,
            base_url=base_url or "https://api.openai.com/v1",
            model=model,
            small_model=model,
            temperature=float(os.environ.get("OPENAI_TEMPERATURE", "0.2")),
            max_tokens=int(os.environ.get("OPENAI_MAX_TOKENS", "4096")),
        )
    )
    return llm, HashEmbedder(), SimpleReranker()
