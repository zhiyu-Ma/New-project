from __future__ import annotations

from datetime import datetime, timedelta

from .graphiti_adapter import GraphMemory
from .world import World, episodes_from_world


async def ingest_world(
    memory: GraphMemory,
    world: World,
    start_time: datetime,
    step: timedelta = timedelta(minutes=1),
) -> None:
    current_time = start_time

    for episode in episodes_from_world(world):
        print("\n[ingest] episode name:", episode.name)
        print("[ingest] episode source:", episode.source)
        print("[ingest] episode body:")
        print(episode.body)

        try:
            await memory.add_episode(episode, current_time)
        except Exception as exc:
            print("\n[ingest] 写入失败 episode:", episode.name)
            print("[ingest] source:", episode.source)
            print("[ingest] body:")
            print(episode.body)
            raise

        current_time = current_time + step