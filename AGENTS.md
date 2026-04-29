# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A playable text-adventure game powered by LLM narration, built on top of a [Graphiti](https://github.com/getzep/graphiti) knowledge-graph validation harness. The town of Stonebridge (石桥镇) has 5 NPCs, 4 secrets, and emergent narration driven by dual-context retrieval.

## Starting the game

```bash
. .venv/bin/activate
python scripts/game_server.py
# Open http://127.0.0.1:4173/
```

Requires `OPENAI_API_KEY` (and optionally `OPENAI_BASE_URL`, `OPENAI_MODEL`) in `.env`.

## Commands

```bash
# Start the playable web server
python scripts/game_server.py

# Run all local checks (unit tests + syntax + world validation)
scripts/check.sh

# Run unit tests directly
python3 -m unittest discover -s tests -v

# Run a single test file
python3 -m unittest tests.test_engine -v

# Syntax-check all source modules
python3 -m compileall graphiti_text_adventure

# Print the world-generation prompt (for feeding to an LLM)
python3 -m graphiti_text_adventure.cli init-prompt

# Validate a world JSON file
python3 -m graphiti_text_adventure.cli validate-world examples/stonebridge_world.json

# Print initialization episodes as JSONL
python3 -m graphiti_text_adventure.cli episodes examples/stonebridge_world.json
```

### Real Graphiti / Neo4j smoke test (requires credentials)

```bash
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=...
export OPENAI_API_KEY=...
zsh -lic '. .venv/bin/activate && python scripts/real_graphiti_smoke.py'
```

## Architecture

### Data flow

```
World JSON → validate_world_data() → world_from_dict() → episodes_from_world()
         → ingest_world() → GraphMemory.add_episode() (one episode per entity)

Per turn:
  PlayerAction + TurnScene
    → retrieve_dual_context()            # two parallel memory.search() + two graph traversals
    → build_narration_prompt()           # formats DualContext into LLM prompt
    → Narrator.narrate()                 # returns {decision, narrative}
    → memory.add_episode() × 2          # action + narrative stored back
```

### Key modules

| Module | Role |
|---|---|
| `world.py` | `World` dataclasses, `validate_world_data()` (enforces 5 NPCs, 6+ relationships, 4+ secrets, cross-links), `episodes_from_world()` (converts World → Chinese-language `EpisodeDraft` list) |
| `graphiti_adapter.py` | `GraphMemory` Protocol + `GraphitiMemory` real impl. Keeps `graphiti-core` optional — tests run offline against stub implementations. Also defines the Pydantic ontology (Person/Location/Topic/edge types) passed to Graphiti. |
| `retrieval.py` | `retrieve_dual_context()`: fires four async queries (player-view search, entity graph traversal, location traversal, semantic search) and merges them into `DualContext`. `CypherTraversalProvider` runs raw Cypher against Neo4j. |
| `engine.py` | `GameEngine.play_turn()` orchestrates one game turn end-to-end. `GameClock` advances a simulated timestamp for each `add_episode` call. |
| `prompts.py` | Builds the narration prompt string from `DualContext`. |
| `narrator.py` | `Narrator` Protocol implementations (stub + real LLM). |
| `runtime_clients.py` | Constructs `graphiti-core` LLM/embedder/cross-encoder clients from env vars. |
| `models.py` | Shared frozen dataclasses: `EpisodeDraft`, `PlayerAction`, `NarrationDecision`. |
| `ingestion.py` | Thin `ingest_world()` async loop over `episodes_from_world()` output. |
| `cli.py` | Typer CLI wrapping `init-prompt`, `validate-world`, `episodes` subcommands. |

### Protocol pattern

`GraphMemory` and `GraphTraversalProvider` are structural `Protocol` types. Tests inject lightweight stubs; production code uses `GraphitiMemory` / `CypherTraversalProvider`. No mocking library is needed — just implement the two methods.

### World constraints (enforced by `validate_world_data`)

- Exactly 5 NPCs
- ≥ 6 NPC relationships
- ≥ 4 secrets, ≥ 2 involving multiple NPCs
- ≥ 1 secret linked to a public event, ≥ 1 secret cross-linked to another secret
- All `id` cross-references (regulars, involved_npcs, knowers, related_to) must resolve

### Episode body language

All `episodes_from_world()` output is in Chinese (Mandarin). The retrieval query strings in `retrieval.py` are also in Chinese. This is intentional — the demo targets a Chinese-language narrative setting.
