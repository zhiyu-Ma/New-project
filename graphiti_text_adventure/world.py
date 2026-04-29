from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .models import EpisodeDraft, EpisodeSource


class WorldValidationError(ValueError):
    def __init__(self, issues: list[str]):
        super().__init__("world validation failed: " + "; ".join(issues))
        self.issues = issues


@dataclass(frozen=True)
class WorldQualityReport:
    npc_count: int
    relationship_count: int
    secret_count: int
    multi_person_secret_count: int
    secret_event_link_count: int
    secret_cross_link_count: int


@dataclass(frozen=True)
class Setting:
    town_name: str
    era: str
    atmosphere: str


@dataclass(frozen=True)
class NPC:
    id: str
    full_name: str
    occupation: str
    personality_tags: list[str]
    current_state: str
    motivation: str


@dataclass(frozen=True)
class Location:
    id: str
    name: str
    description: str
    regulars: list[str]


@dataclass(frozen=True)
class PublicEvent:
    id: str
    description: str
    involved_npcs: list[str]
    when: str


@dataclass(frozen=True)
class Secret:
    id: str
    description: str
    knowers: list[str]
    truth_value: str
    severity: str
    related_to: list[str]


@dataclass(frozen=True)
class NPCRelationship:
    from_id: str
    to_id: str
    type: str
    origin: str


@dataclass(frozen=True)
class ProtagonistLink:
    npc_id: str
    familiarity: str


@dataclass(frozen=True)
class Protagonist:
    background: str
    knows_npcs: list[ProtagonistLink]
    knows_events: list[str]
    knows_secrets: list[str]


@dataclass(frozen=True)
class World:
    setting: Setting
    npcs: list[NPC]
    locations: list[Location]
    public_events: list[PublicEvent]
    secrets: list[Secret]
    npc_relationships: list[NPCRelationship]
    protagonist: Protagonist


def validate_world_data(data: dict[str, Any]) -> WorldQualityReport:
    npcs = data.get("npcs", [])
    relationships = data.get("npc_relationships", [])
    secrets = data.get("secrets", [])
    events = data.get("public_events", [])
    event_ids = {event.get("id") for event in events}

    multi_person_secret_count = sum(1 for secret in secrets if len(secret.get("knowers", [])) > 1)
    secret_event_link_count = sum(
        1
        for secret in secrets
        if any(related_id in event_ids for related_id in secret.get("related_to", []))
    )
    secret_cross_link_count = sum(
        1
        for secret in secrets
        for related_id in secret.get("related_to", [])
        if isinstance(related_id, str) and related_id.startswith("secret_")
    )

    report = WorldQualityReport(
        npc_count=len(npcs),
        relationship_count=len(relationships),
        secret_count=len(secrets),
        multi_person_secret_count=multi_person_secret_count,
        secret_event_link_count=secret_event_link_count,
        secret_cross_link_count=secret_cross_link_count,
    )

    issues: list[str] = []
    if report.npc_count != 5:
        issues.append("exactly 5 NPCs")
    if report.relationship_count < 6:
        issues.append("at least 6 NPC relationships")
    if report.secret_count < 4:
        issues.append("at least 4 secrets")
    if report.multi_person_secret_count < 2:
        issues.append("at least 2 secrets involving multiple NPCs")
    if report.secret_event_link_count < 1:
        issues.append("at least 1 secret linked to a public event")
    if report.secret_cross_link_count < 1:
        issues.append("at least 1 cross-link between secrets")

    issues.extend(_reference_issues(data))
    if issues:
        raise WorldValidationError(issues)

    return report


def world_from_dict(data: dict[str, Any]) -> World:
    validate_world_data(data)
    setting = data["setting"]
    return World(
        setting=Setting(
            town_name=setting["town_name"],
            era=setting["era"],
            atmosphere=setting["atmosphere"],
        ),
        npcs=[
            NPC(
                id=item["id"],
                full_name=item["full_name"],
                occupation=item["occupation"],
                personality_tags=list(item["personality_tags"]),
                current_state=item["current_state"],
                motivation=item["motivation"],
            )
            for item in data["npcs"]
        ],
        locations=[
            Location(
                id=item["id"],
                name=item["name"],
                description=item["description"],
                regulars=list(item["regulars"]),
            )
            for item in data["locations"]
        ],
        public_events=[
            PublicEvent(
                id=item["id"],
                description=item["description"],
                involved_npcs=list(item["involved_npcs"]),
                when=item["when"],
            )
            for item in data["public_events"]
        ],
        secrets=[
            Secret(
                id=item["id"],
                description=item["description"],
                knowers=list(item["knowers"]),
                truth_value=item["truth_value"],
                severity=item["severity"],
                related_to=list(item.get("related_to", [])),
            )
            for item in data["secrets"]
        ],
        npc_relationships=[
            NPCRelationship(
                from_id=item["from"],
                to_id=item["to"],
                type=item["type"],
                origin=item["origin"],
            )
            for item in data["npc_relationships"]
        ],
        protagonist=Protagonist(
            background=data["protagonist"]["background"],
            knows_npcs=[
                ProtagonistLink(npc_id=item["npc_id"], familiarity=item["familiarity"])
                for item in data["protagonist"].get("knows_npcs", [])
            ],
            knows_events=list(data["protagonist"].get("knows_events", [])),
            knows_secrets=list(data["protagonist"].get("knows_secrets", [])),
        ),
    )


def episodes_from_world(world: World) -> list[EpisodeDraft]:
    names = _npc_names(world)
    event_descriptions = {event.id: event.description for event in world.public_events}
    secret_descriptions = {secret.id: secret.description for secret in world.secrets}
    episodes: list[EpisodeDraft] = []

    # ─── NPC 背景：拆成多条简单事实，避免 Graphiti 抽成嵌套 profile map ───
    for npc in world.npcs:
        full_name = npc.full_name
        occupation = _strip_terminal_punctuation(npc.occupation)
        personality = _join(npc.personality_tags)
        current_state = _strip_terminal_punctuation(npc.current_state)
        motivation = _strip_terminal_punctuation(npc.motivation)

        episodes.append(
            _text_episode(
                f"{npc.id}_occupation",
                f"{full_name}在镇上从事{occupation}这个营生。",
            )
        )

        if personality:
            episodes.append(
                _text_episode(
                    f"{npc.id}_personality",
                    f"镇上的人通常觉得{full_name}{personality}。",
                )
            )

        if current_state:
            episodes.append(
                _text_episode(
                    f"{npc.id}_current_state",
                    f"最近，{full_name}{current_state}。",
                )
            )

        if motivation:
            episodes.append(
                _text_episode(
                    f"{npc.id}_motivation",
                    f"{full_name}目前最在意的事情是{motivation}。",
                )
            )

    # ─── 地点：拆成描述和常驻人物 ───
    for location in world.locations:
        location_desc = _strip_terminal_punctuation(location.description)

        episodes.append(
            _text_episode(
                f"{location.id}_description",
                f"{location.name}是镇上的一处地点。{location_desc}。",
            )
        )

        regular_names = [names[npc_id] for npc_id in location.regulars if npc_id in names]
        if regular_names:
            regulars = _join(regular_names)
            episodes.append(
                _text_episode(
                    f"{location.id}_regulars",
                    f"{regulars}经常出现在{location.name}。",
                )
            )

    # ─── 公开事件 ───
    for event in world.public_events:
        event_desc = _strip_terminal_punctuation(event.description)
        involved_names = [names[npc_id] for npc_id in event.involved_npcs if npc_id in names]

        episodes.append(
            _text_episode(
                f"{event.id}_description",
                f"{event.when}，镇上发生了这件事：{event_desc}。",
            )
        )

        if involved_names:
            involved = _join(involved_names)
            episodes.append(
                _text_episode(
                    f"{event.id}_involved",
                    f"{involved}和{event_desc}这件事有关。",
                )
            )

    # ─── 秘密：避免“真实性：”“严重程度：”这种字段式写法 ───
    for secret in world.secrets:
        secret_desc = _strip_terminal_punctuation(secret.description)
        knower_names = [names[npc_id] for npc_id in secret.knowers if npc_id in names]
        knowers = _join(knower_names)

        if knowers:
            episodes.append(
                _text_episode(
                    f"{secret.id}_knowers",
                    f"{knowers}知道一件隐秘的事情：{secret_desc}。",
                )
            )
        else:
            episodes.append(
                _text_episode(
                    f"{secret.id}_description",
                    f"镇上存在一件隐秘的事情：{secret_desc}。",
                )
            )

        if secret.truth_value:
            episodes.append(
                _text_episode(
                    f"{secret.id}_truth",
                    f"关于“{secret_desc}”这件事，它的真实情况是{_strip_terminal_punctuation(secret.truth_value)}。",
                )
            )

        if secret.severity:
            episodes.append(
                _text_episode(
                    f"{secret.id}_severity",
                    f"“{secret_desc}”这件事在镇上的影响程度属于{secret.severity}。",
                )
            )

        related = _related_descriptions(secret.related_to, event_descriptions, secret_descriptions)
        if related:
            episodes.append(
                _text_episode(
                    f"{secret.id}_related",
                    f"“{secret_desc}”这件事和这些情况有关：{related}。",
                )
            )

    # ─── NPC 关系 ───
    for relationship in world.npc_relationships:
        from_name = names[relationship.from_id]
        to_name = names[relationship.to_id]
        rel_type = _strip_terminal_punctuation(relationship.type)
        origin = _strip_terminal_punctuation(relationship.origin)

        episodes.append(
            _text_episode(
                f"relationship_{relationship.from_id}_{relationship.to_id}",
                f"{from_name}对{to_name}抱有{rel_type}的态度，这种态度来自于{origin}。",
            )
        )

    # ─── 主角背景 ───
    episodes.append(
        _text_episode(
            "protagonist_background",
            _protagonist_background_body(world.protagonist.background),
        )
    )

    # ─── 主角认识的 NPC ───
    for link in world.protagonist.knows_npcs:
        npc_name = names[link.npc_id]
        familiarity = _strip_terminal_punctuation(link.familiarity)

        episodes.append(
            _text_episode(
                f"protagonist_knows_{link.npc_id}",
                f"主角认识{npc_name}，两人的熟悉情况是{familiarity}。",
            )
        )

    # ─── 主角知道的公开事件 ───
    for event_id in world.protagonist.knows_events:
        event_desc = _strip_terminal_punctuation(event_descriptions[event_id])

        episodes.append(
            _text_episode(
                f"protagonist_knows_{event_id}",
                f"主角知道镇上发生过这件公开事件：{event_desc}。",
            )
        )

    # ─── 主角知道的秘密 ───
    for secret_id in world.protagonist.knows_secrets:
        secret_desc = _strip_terminal_punctuation(secret_descriptions[secret_id])

        episodes.append(
            _text_episode(
                f"protagonist_knows_{secret_id}",
                f"主角知道一件秘密：{secret_desc}。",
            )
        )

    return episodes

def _text_episode(name: str, body: str) -> EpisodeDraft:
    return EpisodeDraft(
        name=name,
        body=body,
        source=EpisodeSource.TEXT,
        source_description="initial world seed",
    )


def _npc_names(world: World) -> dict[str, str]:
    return {npc.id: npc.full_name for npc in world.npcs}


def _join(items: list[str]) -> str:
    return "、".join(items)


def _related_descriptions(
    related_ids: list[str],
    event_descriptions: dict[str, str],
    secret_descriptions: dict[str, str],
) -> str:
    descriptions: list[str] = []
    for related_id in related_ids:
        if related_id in event_descriptions:
            descriptions.append(_strip_terminal_punctuation(event_descriptions[related_id]))
        elif related_id in secret_descriptions:
            descriptions.append(_strip_terminal_punctuation(secret_descriptions[related_id]))
    return _join(descriptions)


def _protagonist_background_body(background: str) -> str:
    normalized = _strip_terminal_punctuation(background)
    if normalized.startswith("主角"):
        return f"{normalized}。"
    return f"主角是{normalized}。"


def _strip_terminal_punctuation(text: str) -> str:
    return text.rstrip("。.!！?？")


def _reference_issues(data: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    npc_ids = {npc.get("id") for npc in data.get("npcs", [])}
    event_ids = {event.get("id") for event in data.get("public_events", [])}
    secret_ids = {secret.get("id") for secret in data.get("secrets", [])}

    for location in data.get("locations", []):
        for npc_id in location.get("regulars", []):
            if npc_id not in npc_ids:
                issues.append(f"unknown npc id {npc_id} in location {location.get('id')} regulars")

    for event in data.get("public_events", []):
        for npc_id in event.get("involved_npcs", []):
            if npc_id not in npc_ids:
                issues.append(f"unknown npc id {npc_id} in event {event.get('id')} involved_npcs")

    for secret in data.get("secrets", []):
        for npc_id in secret.get("knowers", []):
            if npc_id not in npc_ids:
                issues.append(f"unknown npc id {npc_id} in secret {secret.get('id')} knowers")
        for related_id in secret.get("related_to", []):
            if related_id not in event_ids and related_id not in secret_ids:
                issues.append(f"unknown related id {related_id} in secret {secret.get('id')}")

    for relationship in data.get("npc_relationships", []):
        for field in ("from", "to"):
            npc_id = relationship.get(field)
            if npc_id not in npc_ids:
                issues.append(f"unknown npc id {npc_id} in relationship {relationship.get('from')}->{relationship.get('to')}")

    protagonist = data.get("protagonist", {})
    for link in protagonist.get("knows_npcs", []):
        npc_id = link.get("npc_id")
        if npc_id not in npc_ids:
            issues.append(f"unknown npc id {npc_id} in protagonist knows_npcs")
    for event_id in protagonist.get("knows_events", []):
        if event_id not in event_ids:
            issues.append(f"unknown event id {event_id} in protagonist knows_events")
    for secret_id in protagonist.get("knows_secrets", []):
        if secret_id not in secret_ids:
            issues.append(f"unknown secret id {secret_id} in protagonist knows_secrets")

    return issues
