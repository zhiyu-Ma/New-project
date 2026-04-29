from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Protocol

from .graphiti_adapter import GraphMemory, MemoryEdge
from .models import PlayerAction


@dataclass(frozen=True)
class TopicCandidate:
    topic_id: str
    name: str
    description: str
    topic_type: str
    truth_value: str
    known_by_protagonist: bool
    last_seen_turn: int | None
    source: str
    knowers: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class DualContext:
    player_view: list[MemoryEdge]
    god_view_topics: list[TopicCandidate]


class GraphTraversalProvider(Protocol):
    async def topics_near_entities(self, entity_names: list[str], hops: int = 2) -> list[TopicCandidate]:
        ...

    async def topics_near_location(self, location_name: str) -> list[TopicCandidate]:
        ...


class EmptyTraversalProvider:
    async def topics_near_entities(self, entity_names: list[str], hops: int = 2) -> list[TopicCandidate]:
        return []

    async def topics_near_location(self, location_name: str) -> list[TopicCandidate]:
        return []


QueryRunner = Callable[[str, dict[str, Any]], Awaitable[list[dict[str, Any]]]]


def query_runner_from_neo4j_driver(driver: Any) -> QueryRunner:
    async def run(query: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        records, _, _ = await driver.execute_query(query, **params)
        return [dict(record) for record in records]

    return run


class CypherTraversalProvider:
    def __init__(self, query_runner: QueryRunner, protagonist_name: str = "主角"):
        self._query_runner = query_runner
        self._protagonist_name = protagonist_name

    async def topics_near_entities(self, entity_names: list[str], hops: int = 2) -> list[TopicCandidate]:
        if hops not in (1, 2):
            raise ValueError("hops must be 1 or 2 for demo traversal")
        query = f"""
MATCH (start)
WHERE start.name IN $entity_names
MATCH (start)-[*1..{hops}]-(topic)
WHERE 'Topic' IN labels(topic)
OPTIONAL MATCH (protagonist {{name: $protagonist_name}})-[]-(topic)
RETURN DISTINCT
  coalesce(topic.uuid, topic.id, elementId(topic)) AS topic_id,
  topic.name AS name,
  coalesce(topic.description, topic.summary, '') AS description,
  coalesce(topic.topic_type, topic.attributes.topic_type, 'unknown') AS topic_type,
  coalesce(topic.truth_value, topic.attributes.truth_value, 'unknown') AS truth_value,
  protagonist IS NOT NULL AS known_by_protagonist,
  topic.last_seen_turn AS last_seen_turn,
  coalesce(topic.knowers, []) AS knowers
""".strip()
        rows = await self._query_runner(
            query,
            {"entity_names": entity_names, "protagonist_name": self._protagonist_name},
        )
        return [_topic_from_row(row, source="entity") for row in rows]

    async def topics_near_location(self, location_name: str) -> list[TopicCandidate]:
        query = """
MATCH (location)
WHERE location.name = $location_name
MATCH (location)-[*1..2]-(topic)
WHERE 'Topic' IN labels(topic)
OPTIONAL MATCH (protagonist {name: $protagonist_name})-[]-(topic)
RETURN DISTINCT
  coalesce(topic.uuid, topic.id, elementId(topic)) AS topic_id,
  topic.name AS name,
  coalesce(topic.description, topic.summary, '') AS description,
  coalesce(topic.topic_type, topic.attributes.topic_type, 'unknown') AS topic_type,
  coalesce(topic.truth_value, topic.attributes.truth_value, 'unknown') AS truth_value,
  protagonist IS NOT NULL AS known_by_protagonist,
  topic.last_seen_turn AS last_seen_turn,
  coalesce(topic.knowers, []) AS knowers
""".strip()
        rows = await self._query_runner(
            query,
            {"location_name": location_name, "protagonist_name": self._protagonist_name},
        )
        return [_topic_from_row(row, source="location") for row in rows]


def build_player_view_query(
    protagonist_name: str,
    location_name: str,
    present_people: list[str],
    action: PlayerAction,
) -> str:
    target_text = action.target or "无对象"
    modifier_text = action.modifier or "无"
    return (
        f"{protagonist_name}在{location_name}对{target_text}执行{action.verb}，"
        f"话题或修饰为{modifier_text}。检索主角已知的人、地点、事件、传闻、秘密和物品。"
    )


def build_semantic_topic_query(location_name: str, present_people: list[str], action: PlayerAction) -> str:
    people = "、".join(present_people) if present_people else "无"
    return (
        f"{location_name}；在场人物：{people}；玩家动作：{action.to_query_text()}；"
        "检索潜在相关的事件、秘密、传闻、物品 Topic。"
    )


async def retrieve_dual_context(
    memory: GraphMemory,
    traversal: GraphTraversalProvider,
    protagonist_name: str,
    location_name: str,
    present_people: list[str],
    action: PlayerAction,
    turn_number: int,
    topic_limit: int = 5,
) -> DualContext:
    player_query = build_player_view_query(
        protagonist_name=protagonist_name,
        location_name=location_name,
        present_people=present_people,
        action=action,
    )
    semantic_query = build_semantic_topic_query(location_name, present_people, action)
    entity_names = _dedupe([*present_people, action.target, location_name])

    player_task = memory.search(player_query)
    entity_task = traversal.topics_near_entities(entity_names, hops=2)
    location_task = traversal.topics_near_location(location_name)
    semantic_task = memory.search(semantic_query)

    player_view, entity_topics, location_topics, semantic_edges = await asyncio.gather(
        player_task,
        entity_task,
        location_task,
        semantic_task,
    )
    semantic_topics = [_topic_from_semantic_edge(edge) for edge in semantic_edges]
    candidates = _dedupe_topics([*entity_topics, *location_topics, *semantic_topics])
    return DualContext(
        player_view=player_view,
        god_view_topics=rank_topics(candidates, current_turn=turn_number, limit=topic_limit),
    )


def rank_topics(
    candidates: list[TopicCandidate],
    current_turn: int,
    limit: int = 5,
) -> list[TopicCandidate]:
    return sorted(candidates, key=lambda topic: _topic_score(topic, current_turn), reverse=True)[:limit]


def _topic_score(topic: TopicCandidate, current_turn: int) -> float:
    type_score = {
        "secret": 40,
        "rumor": 30,
        "event": 20,
        "item": 10,
    }.get(topic.topic_type, 0)
    known_score = -30 if topic.known_by_protagonist else 50
    if topic.last_seen_turn is None:
        staleness_score = 20
    else:
        staleness_score = max(0, min(current_turn - topic.last_seen_turn, 10))
    source_score = {
        "entity": 4,
        "location": 3,
        "semantic": 2,
    }.get(topic.source, 0)
    return known_score + type_score + staleness_score + source_score


def _dedupe(items: list[str | None]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        if item and item not in seen:
            output.append(item)
            seen.add(item)
    return output


def _dedupe_topics(candidates: list[TopicCandidate]) -> list[TopicCandidate]:
    by_id: dict[str, TopicCandidate] = {}
    for candidate in candidates:
        by_id.setdefault(candidate.topic_id, candidate)
    return list(by_id.values())


def _topic_from_row(row: dict[str, Any], source: str) -> TopicCandidate:
    return TopicCandidate(
        topic_id=str(row.get("topic_id")),
        name=str(row.get("name") or row.get("topic_id")),
        description=str(row.get("description") or ""),
        topic_type=str(row.get("topic_type") or "unknown"),
        truth_value=str(row.get("truth_value") or "unknown"),
        known_by_protagonist=bool(row.get("known_by_protagonist")),
        last_seen_turn=row.get("last_seen_turn"),
        source=source,
        knowers=list(row.get("knowers") or []),
    )


def _topic_from_semantic_edge(edge: MemoryEdge) -> TopicCandidate:
    attributes = edge.attributes or {}
    topic_id = str(attributes.get("topic_id") or edge.uuid or edge.fact)
    name = str(attributes.get("name") or _semantic_topic_name(edge.fact, topic_id))
    return TopicCandidate(
        topic_id=topic_id,
        name=name,
        description=edge.fact,
        topic_type=str(attributes.get("topic_type") or _infer_topic_type(edge.fact)),
        truth_value=str(attributes.get("truth_value") or "unknown"),
        known_by_protagonist=bool(attributes.get("known_by_protagonist", False)),
        last_seen_turn=attributes.get("last_seen_turn"),
        source="semantic",
        knowers=list(attributes.get("knowers") or []),
    )


def _semantic_topic_name(fact: str, fallback: str) -> str:
    stripped = fact.strip()
    if not stripped:
        return fallback
    return _strip_topic_prefix(stripped).rstrip("。.!！?？")[:40]


def _infer_topic_type(fact: str) -> str:
    if "秘密" in fact:
        return "secret"
    if "传闻" in fact:
        return "rumor"
    if "事件" in fact:
        return "event"
    if "物品" in fact or "字据" in fact or "账本" in fact:
        return "item"
    return "unknown"


def _strip_topic_prefix(text: str) -> str:
    for prefix in ("秘密：", "秘密:", "传闻：", "传闻:", "事件：", "事件:", "物品：", "物品:"):
        if text.startswith(prefix):
            return text[len(prefix) :]
    return text
