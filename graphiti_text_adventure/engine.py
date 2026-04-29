from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Protocol

from .graphiti_adapter import GraphMemory
from .models import EpisodeDraft, EpisodeSource, NarrationDecision, PlayerAction
from .prompts import build_narration_prompt
from .retrieval import DualContext, GraphTraversalProvider, retrieve_dual_context


class Narrator(Protocol):
    async def narrate(self, prompt: str) -> dict[str, Any] | str:
        ...


@dataclass(frozen=True)
class TurnScene:
    location_name: str
    present_people: list[str]
    turn_number: int
    day: int
    time_label: str


@dataclass(frozen=True)
class TurnResult:
    decision: NarrationDecision
    narrative: str
    context: DualContext


class GameClock:
    def __init__(self, start: datetime, step: timedelta = timedelta(minutes=1)):
        self._current = start
        self._step = step

    def stamp(self) -> datetime:
        value = self._current
        self._current = self._current + self._step
        return value


class GameEngine:
    def __init__(
        self,
        memory: GraphMemory,
        traversal: GraphTraversalProvider,
        narrator: Narrator,
        clock: GameClock,
        protagonist_name: str = "主角",
    ):
        self._memory = memory
        self._traversal = traversal
        self._narrator = narrator
        self._clock = clock
        self._protagonist_name = protagonist_name

    async def play_turn(self, scene: TurnScene, action: PlayerAction) -> TurnResult:
        context = await retrieve_dual_context(
            memory=self._memory,
            traversal=self._traversal,
            protagonist_name=self._protagonist_name,
            location_name=scene.location_name,
            present_people=scene.present_people,
            action=action,
            turn_number=scene.turn_number,
        )
        prompt = build_narration_prompt(
            location_name=scene.location_name,
            present_people=scene.present_people,
            day=scene.day,
            time_label=scene.time_label,
            action=action,
            context=context,
        )
        raw_response = await self._narrator.narrate(prompt)
        response = json.loads(raw_response) if isinstance(raw_response, str) else raw_response
        decision_data = response["decision"]
        decision = NarrationDecision(
            topic_id=decision_data.get("topic_id"),
            reveal_level=decision_data["reveal_level"],
            reasoning=decision_data["reasoning"],
        )
        narrative = response["narrative"]

        action_episode = EpisodeDraft(
            name=f"turn_{scene.turn_number:03d}_player_action",
            body=json.dumps(action.to_dict(), ensure_ascii=False),
            source=EpisodeSource.JSON,
            source_description="player action",
        )
        narrative_episode = EpisodeDraft(
            name=f"turn_{scene.turn_number:03d}_narrative",
            body=narrative,
            source=EpisodeSource.TEXT,
            source_description="narrative result",
        )
        await self._memory.add_episode(action_episode, self._clock.stamp())
        await self._memory.add_episode(narrative_episode, self._clock.stamp())
        return TurnResult(decision=decision, narrative=narrative, context=context)
