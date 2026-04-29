from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graphiti_text_adventure.real_environment import load_project_dotenv, require_real_environment
from graphiti_text_adventure.runtime_clients import build_runtime_clients_from_env


async def main() -> None:
    load_project_dotenv()
    env = require_real_environment()

    try:
        from graphiti_core import Graphiti
        from graphiti_core.nodes import EpisodeType
    except ModuleNotFoundError as error:
        raise RuntimeError("Install graphiti-core to run the real smoke test") from error

    llm_client, embedder, cross_encoder = build_runtime_clients_from_env()
    graphiti = Graphiti(
        env.values["NEO4J_URI"],
        env.values["NEO4J_USER"],
        env.values["NEO4J_PASSWORD"],
        llm_client=llm_client,
        embedder=embedder,
        cross_encoder=cross_encoder,
    )

    group_id = env.values["GRAPHITI_SMOKE_GROUP"]
    await graphiti.add_episode(
        name="llm_probe_episode",
        episode_body="张守义是石桥镇的铁匠铺老板。张守义认识何桂兰，并且张守义不信任何桂兰。",
        source=EpisodeType.text,
        source_description="real graphiti smoke",
        reference_time=datetime(2026, 4, 29, 8, tzinfo=timezone.utc),
        group_id=group_id,
    )
    results = await graphiti.search("张守义和何桂兰是什么关系？", group_ids=[group_id], num_results=5)
    print(f"SEARCH_RESULTS {len(results)}")
    for edge in results:
        print(f"FACT {edge.fact}")

    await graphiti.close()


if __name__ == "__main__":
    asyncio.run(main())
