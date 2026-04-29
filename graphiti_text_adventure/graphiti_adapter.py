from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Protocol

from .models import EpisodeDraft, EpisodeSource


@dataclass(frozen=True)
class MemoryEdge:
    uuid: str | None = None
    fact: str = ""
    source_node_uuid: str | None = None
    target_node_uuid: str | None = None
    valid_at: datetime | None = None
    invalid_at: datetime | None = None
    attributes: dict[str, Any] | None = None

    @classmethod
    def from_graphiti_result(cls, result: Any) -> "MemoryEdge":
        return cls(
            uuid=getattr(result, "uuid", None),
            fact=getattr(result, "fact", str(result)),
            source_node_uuid=getattr(result, "source_node_uuid", None),
            target_node_uuid=getattr(result, "target_node_uuid", None),
            valid_at=getattr(result, "valid_at", None),
            invalid_at=getattr(result, "invalid_at", None),
            attributes=getattr(result, "attributes", None),
        )


class GraphMemory(Protocol):
    async def add_episode(self, episode: EpisodeDraft, reference_time: datetime) -> None:
        ...

    async def search(self, query: str, **kwargs: Any) -> list[MemoryEdge]:
        ...


class GraphitiMemory:
    def __init__(
        self,
        graphiti: Any,
        source_resolver: Callable[[EpisodeSource], Any] | None = None,
        add_episode_options: dict[str, Any] | None = None,
    ):
        self._graphiti = graphiti
        self._source_resolver = source_resolver or _resolve_graphiti_source
        self._add_episode_options = add_episode_options or {}

    async def add_episode(self, episode: EpisodeDraft, reference_time: datetime) -> None:
        await self._graphiti.add_episode(
            name=episode.name,
            episode_body=episode.body,
            source=self._source_resolver(episode.source),
            source_description=episode.source_description,
            reference_time=reference_time,
            **self._add_episode_options,
        )

    async def search(self, query: str, **kwargs: Any) -> list[MemoryEdge]:
        results = await self._graphiti.search(query, **kwargs)
        return [MemoryEdge.from_graphiti_result(result) for result in results]


def _resolve_graphiti_source(source: EpisodeSource) -> Any:
    try:
        from graphiti_core.nodes import EpisodeType
    except ModuleNotFoundError as error:
        raise RuntimeError("graphiti-core is required for real Graphiti runs") from error
    return getattr(EpisodeType, source.value)


async def connect_graphiti_from_env() -> GraphitiMemory:
    import os

    try:
        from graphiti_core import Graphiti
    except ModuleNotFoundError as error:
        raise RuntimeError("Install graphiti-core to connect to Neo4j") from error

    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD")
    if not password:
        raise RuntimeError("NEO4J_PASSWORD must be set")

    from .runtime_clients import build_runtime_clients_from_env

    llm_client, embedder, cross_encoder = build_runtime_clients_from_env()
    graphiti = Graphiti(
        uri,
        user,
        password,
        llm_client=llm_client,
        embedder=embedder,
        cross_encoder=cross_encoder,
    )
    await graphiti.build_indices_and_constraints()
    return GraphitiMemory(graphiti, add_episode_options=build_demo_ontology())


def build_demo_ontology() -> dict[str, Any]:
    try:
        from pydantic import BaseModel
    except ModuleNotFoundError as error:
        raise RuntimeError("pydantic is required to build Graphiti custom entity types") from error

    class Person(BaseModel):
        personality: list[str] | None = None
        current_mood: str | None = None

    class Location(BaseModel):
        description: str | None = None

    class Topic(BaseModel):
        topic_type: str | None = None
        truth_value: str | None = None

    class Relation(BaseModel):
        strength: int | None = None

    edge_types = {
        "KNOWS": Relation,
        "TRUSTS": Relation,
        "DISTRUSTS": Relation,
        "LOCATED_AT": Relation,
        "KNOWS_ABOUT": Relation,
        "INVOLVED_IN": Relation,
    }
    return {
        "entity_types": {
            "Person": Person,
            "Location": Location,
            "Topic": Topic,
        },
        "edge_types": edge_types,
        "edge_type_map": {
            ("Person", "Person"): ["KNOWS", "TRUSTS", "DISTRUSTS"],
            ("Person", "Location"): ["LOCATED_AT"],
            ("Person", "Topic"): ["KNOWS_ABOUT", "INVOLVED_IN"],
            ("Location", "Topic"): ["INVOLVED_IN"],
            ("Topic", "Topic"): ["INVOLVED_IN"],
        },
    }
