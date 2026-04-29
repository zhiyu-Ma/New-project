import asyncio
from datetime import datetime, timezone
from unittest import TestCase

from graphiti_text_adventure.ingestion import ingest_world
from graphiti_text_adventure.models import EpisodeSource
from graphiti_text_adventure.world import world_from_dict
from test_world import sample_world_data


class FakeMemory:
    def __init__(self):
        self.added = []

    async def add_episode(self, episode, reference_time):
        self.added.append((episode, reference_time))


class IngestionTests(TestCase):
    def test_ingest_world_adds_every_episode_with_incrementing_game_time(self):
        memory = FakeMemory()
        world = world_from_dict(sample_world_data())
        start = datetime(2026, 4, 29, 8, tzinfo=timezone.utc)

        asyncio.run(ingest_world(memory, world, start))

        self.assertEqual(len(memory.added), 25)
        self.assertEqual(memory.added[0][0].source, EpisodeSource.TEXT)
        self.assertEqual(memory.added[0][1], start)
        self.assertLess(memory.added[0][1], memory.added[-1][1])
