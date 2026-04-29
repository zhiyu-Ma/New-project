# Graphiti Text Adventure Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python demo harness that validates Graphiti as the memory and retrieval layer for a menu-driven emergent text adventure.

**Architecture:** Keep Graphiti, LLM narration, world validation, episode conversion, retrieval, and turn orchestration behind small interfaces so the demo can be tested without Neo4j or network calls. The real runtime uses Graphiti's `add_episode()` and `search()` APIs, while tests use in-memory fakes.

**Tech Stack:** Python 3.10+, standard-library `unittest`, optional `graphiti-core`, optional OpenAI-compatible HTTP endpoint.

---

### Task 1: World Schema And Episode Conversion

**Files:**
- Create: `graphiti_text_adventure/models.py`
- Create: `graphiti_text_adventure/world.py`
- Test: `tests/test_world.py`

- [ ] Write tests that validate the JSON hard metrics from the proposal.
- [ ] Write tests that convert NPCs, locations, public events, secrets, relationships, and protagonist links into compact natural-language episodes.
- [ ] Implement dataclasses, validation, ID-to-name resolution, and episode conversion.

### Task 2: Graphiti Client Boundary

**Files:**
- Create: `graphiti_text_adventure/graphiti_adapter.py`
- Test: `tests/test_graphiti_adapter.py`

- [ ] Write tests that ingestion passes `reference_time` and correct source values for text and JSON episodes.
- [ ] Implement a protocol-friendly adapter around Graphiti's `add_episode()` and `search()`.
- [ ] Keep `graphiti-core` imports lazy so unit tests run without the dependency installed.

### Task 3: Retrieval And Topic Ranking

**Files:**
- Create: `graphiti_text_adventure/retrieval.py`
- Test: `tests/test_retrieval.py`

- [ ] Write tests for protagonist-view search queries, god-view semantic query construction, and top topic ranking.
- [ ] Implement simple weighted ranking: unknown secrets first, stale topics next, then rumors/events/items.
- [ ] Provide explicit hooks for deterministic graph traversal results so real Cypher can be added without changing orchestration.

### Task 4: Prompting And Turn Orchestration

**Files:**
- Create: `graphiti_text_adventure/prompts.py`
- Create: `graphiti_text_adventure/engine.py`
- Test: `tests/test_engine.py`

- [ ] Write tests that a turn calls dual retrieval, narrates JSON output, displays narrative text, and writes both player action and narrative episodes with game time.
- [ ] Implement the initialization prompt and narration prompt from the proposal with naming and reveal-channel constraints.
- [ ] Implement `GameEngine.play_turn()` over injected `GraphMemory` and `Narrator` ports.

### Task 5: CLI And Documentation

**Files:**
- Create: `graphiti_text_adventure/cli.py`
- Create: `README.md`
- Create: `pyproject.toml`

- [ ] Provide commands for validating a generated world JSON, printing initialization episodes, and running a scripted fake-memory demo.
- [ ] Document the required environment variables for real Graphiti runs: `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, `OPENAI_API_KEY`.
- [ ] Verify all tests with `python3 -m unittest discover -v` and syntax with `python3 -m compileall graphiti_text_adventure`.
