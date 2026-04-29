from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class EpisodeSource(Enum):
    TEXT = "text"
    JSON = "json"
    MESSAGE = "message"


@dataclass(frozen=True)
class EpisodeDraft:
    name: str
    body: str
    source: EpisodeSource
    source_description: str


@dataclass(frozen=True)
class PlayerAction:
    verb: str
    target: str | None = None
    modifier: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "verb": self.verb,
            "target": self.target,
            "modifier": self.modifier,
        }

    def to_query_text(self) -> str:
        parts = [self.verb]
        if self.target:
            parts.append(self.target)
        if self.modifier:
            parts.append(self.modifier)
        return " ".join(parts)

    def to_menu_label(self) -> str:
        parts = [self.verb]
        if self.target:
            parts.append(self.target)
        if self.modifier:
            parts.append(self.modifier)
        return "-".join(parts)


@dataclass(frozen=True)
class NarrationDecision:
    topic_id: str | None
    reveal_level: str
    reasoning: str
