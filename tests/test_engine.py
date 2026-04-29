import asyncio
import json
from datetime import datetime, timezone
from unittest import TestCase

from graphiti_text_adventure.engine import GameClock, GameEngine, TurnScene
from graphiti_text_adventure.graphiti_adapter import MemoryEdge
from graphiti_text_adventure.models import EpisodeSource, PlayerAction
from graphiti_text_adventure.retrieval import GraphTraversalProvider, TopicCandidate


class FakeMemory:
    def __init__(self):
        self.added = []

    async def search(self, query, **kwargs):
        return [MemoryEdge(uuid="edge-1", fact="主角认识张守义。")]

    async def add_episode(self, episode, reference_time):
        self.added.append((episode, reference_time))


class FakeTraversal(GraphTraversalProvider):
    async def topics_near_entities(self, entity_names, hops=2):
        return [
            TopicCandidate(
                topic_id="secret_01",
                name="高息货款",
                description="张守义欠何桂兰一大笔高息货款。",
                topic_type="secret",
                truth_value="true",
                known_by_protagonist=False,
                last_seen_turn=None,
                source="entity",
                knowers=["张守义", "何桂兰"],
            )
        ]

    async def topics_near_location(self, location_name):
        return []


class FakeNarrator:
    async def narrate(self, prompt):
        self.prompt = prompt
        return {
            "decision": {
                "topic_id": "secret_01",
                "reveal_level": "hint",
                "reasoning": "玩家询问财务状况，秘密高度相关。",
            },
            "narrative": "张守义听见“财务状况”四个字，手里的铁钳磕在炉沿上。他很快笑了笑，说铺子还撑得住，却把柜台下一张写着何桂兰名字的字条往里推了推。",
        }


class EngineTests(TestCase):
    def test_play_turn_retrieves_narrates_and_writes_action_and_narrative_episodes(self):
        memory = FakeMemory()
        narrator = FakeNarrator()
        engine = GameEngine(
            memory=memory,
            traversal=FakeTraversal(),
            narrator=narrator,
            clock=GameClock(start=datetime(2026, 4, 29, 8, tzinfo=timezone.utc)),
        )
        scene = TurnScene(
            location_name="张守义铁匠铺",
            present_people=["张守义", "何桂兰"],
            turn_number=1,
            day=1,
            time_label="上午",
        )
        action = PlayerAction(verb="打听", target="张守义", modifier="财务状况")

        result = asyncio.run(engine.play_turn(scene, action))

        self.assertEqual(result.decision.topic_id, "secret_01")
        self.assertIn("柜台下一张写着何桂兰名字的字条", result.narrative)
        self.assertIn("# 剧情燃料", narrator.prompt)
        self.assertIn("[secret_01] 高息货款：张守义欠何桂兰一大笔高息货款。", narrator.prompt)
        self.assertEqual(len(memory.added), 2)
        action_episode, action_time = memory.added[0]
        narrative_episode, narrative_time = memory.added[1]
        self.assertEqual(action_episode.source, EpisodeSource.JSON)
        self.assertEqual(json.loads(action_episode.body)["modifier"], "财务状况")
        self.assertEqual(narrative_episode.source, EpisodeSource.TEXT)
        self.assertIn("张守义听见", narrative_episode.body)
        self.assertLess(action_time, narrative_time)
