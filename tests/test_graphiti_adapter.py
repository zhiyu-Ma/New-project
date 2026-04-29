import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest import TestCase

from graphiti_text_adventure.graphiti_adapter import GraphitiMemory, MemoryEdge
from graphiti_text_adventure.models import EpisodeDraft, EpisodeSource


class FakeGraphiti:
    def __init__(self):
        self.added = []
        self.searches = []

    async def add_episode(self, **kwargs):
        self.added.append(kwargs)

    async def search(self, query, **kwargs):
        self.searches.append((query, kwargs))
        return [
            SimpleNamespace(
                uuid="edge-1",
                fact="张守义 distrusts 何桂兰",
                source_node_uuid="person-1",
                target_node_uuid="person-2",
                valid_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
                invalid_at=None,
            )
        ]


class GraphitiMemoryTests(TestCase):
    def test_add_episode_preserves_reference_time_and_source(self):
        fake = FakeGraphiti()
        memory = GraphitiMemory(fake, source_resolver=lambda source: source.value)
        episode = EpisodeDraft(
            name="init_npc_01",
            body="张守义是铁匠铺老板。",
            source=EpisodeSource.TEXT,
            source_description="initial world seed",
        )
        reference_time = datetime(2026, 4, 29, 8, tzinfo=timezone.utc)

        asyncio.run(memory.add_episode(episode, reference_time))

        self.assertEqual(
            fake.added,
            [
                {
                    "name": "init_npc_01",
                    "episode_body": "张守义是铁匠铺老板。",
                    "source": "text",
                    "source_description": "initial world seed",
                    "reference_time": reference_time,
                }
            ],
        )

    def test_add_episode_can_pass_demo_ontology_options(self):
        fake = FakeGraphiti()
        options = {"entity_types": {"Person": object}, "edge_types": {"TRUSTS": object}}
        memory = GraphitiMemory(fake, source_resolver=lambda source: source.value, add_episode_options=options)
        episode = EpisodeDraft(
            name="init_npc_01",
            body="张守义是铁匠铺老板。",
            source=EpisodeSource.TEXT,
            source_description="initial world seed",
        )
        reference_time = datetime(2026, 4, 29, 8, tzinfo=timezone.utc)

        asyncio.run(memory.add_episode(episode, reference_time))

        self.assertEqual(fake.added[0]["entity_types"], {"Person": object})
        self.assertEqual(fake.added[0]["edge_types"], {"TRUSTS": object})

    def test_search_normalizes_graphiti_edge_results(self):
        fake = FakeGraphiti()
        memory = GraphitiMemory(fake, source_resolver=lambda source: source.value)

        edges = asyncio.run(memory.search("主角知道什么", center_node_uuid="person-main", limit=5))

        self.assertEqual(fake.searches, [("主角知道什么", {"center_node_uuid": "person-main", "limit": 5})])
        self.assertEqual(
            edges,
            [
                MemoryEdge(
                    uuid="edge-1",
                    fact="张守义 distrusts 何桂兰",
                    source_node_uuid="person-1",
                    target_node_uuid="person-2",
                    valid_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
                    invalid_at=None,
                )
            ],
        )
