#!/usr/bin/env python3
"""Playable game server — connects the web frontend to real LLM narration.

Usage:
    . .venv/bin/activate
    python scripts/game_server.py

Then open http://127.0.0.1:4173/

Optional env vars (in .env):
    OPENAI_API_KEY       required
    OPENAI_BASE_URL      default: https://api.openai.com/v1
    OPENAI_MODEL         default: gpt-4.1-mini
    NEO4J_URI            default: bolt://localhost:7687
    NEO4J_USER           default: neo4j
    NEO4J_PASSWORD       if set → enables Graphiti/Neo4j mode
    SKIP_WORLD_INGEST    set to 1 to skip world re-ingestion (use existing graph)
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ─── Env loading (manual fallback so python-dotenv is optional) ───────────────

def _load_dotenv(path: Path) -> None:
    if not path.exists():
        print(f"[env] .env 文件不存在: {path}")
        return
    try:
        from dotenv import load_dotenv
        load_dotenv(path, override=False)
        print(f"[env] 已用 python-dotenv 加载: {path}")
        return
    except ImportError:
        pass
    # Manual parser fallback
    loaded = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val
                loaded += 1
    print(f"[env] 手动解析 .env，写入 {loaded} 个变量")

_load_dotenv(ROOT / ".env")

# ─── Print resolved env for debugging ────────────────────────────────────────

def _masked(v: str) -> str:
    return v[:4] + "****" if len(v) > 4 else "****"

print("[env] OPENAI_API_KEY  =", _masked(os.environ.get("OPENAI_API_KEY", "")))
print("[env] OPENAI_BASE_URL =", os.environ.get("OPENAI_BASE_URL", "(default openai)"))
print("[env] OPENAI_MODEL    =", os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"))
print("[env] NEO4J_URI       =", os.environ.get("NEO4J_URI", "bolt://localhost:7687"))
print("[env] NEO4J_PASSWORD  =", "set" if os.environ.get("NEO4J_PASSWORD") else "not set")
print()

from graphiti_text_adventure.graphiti_adapter import MemoryEdge
from graphiti_text_adventure.models import EpisodeDraft, EpisodeSource, PlayerAction
from graphiti_text_adventure.narrator import OpenAICompatibleNarrator
from graphiti_text_adventure.prompts import build_narration_prompt
from graphiti_text_adventure.retrieval import (
    DualContext, EmptyTraversalProvider, TopicCandidate,
    CypherTraversalProvider, query_runner_from_neo4j_driver, retrieve_dual_context,
)
from graphiti_text_adventure.world import World, world_from_dict

PORT = 4173
WEB_ROOT = ROOT / "web"
TIME_LABELS = ["上午", "中午", "傍晚", "夜里"]

# ─── Dedicated async event loop (keeps Neo4j connections alive) ───────────────

_async_loop = asyncio.new_event_loop()
_async_thread = threading.Thread(target=_async_loop.run_forever, daemon=True)
_async_thread.start()


def run_async(coro, timeout: int = 300):
    return asyncio.run_coroutine_threadsafe(coro, _async_loop).result(timeout=timeout)


# ─── Optional Graphiti / Neo4j resources ─────────────────────────────────────

_graphiti_memory = None   # GraphitiMemory | None
_traversal = None         # CypherTraversalProvider | None
_neo4j_driver = None
_graphiti_clock: datetime = datetime(2024, 6, 1, 8, 0, 0)
_graphiti_clock_step = timedelta(minutes=2)
_graphiti_clock_lock = threading.Lock()


def _next_game_time() -> datetime:
    global _graphiti_clock
    with _graphiti_clock_lock:
        t = _graphiti_clock
        _graphiti_clock += _graphiti_clock_step
    return t


def _init_graphiti(world: World) -> None:
    global _graphiti_memory, _traversal, _neo4j_driver
    neo4j_password = os.environ.get("NEO4J_PASSWORD", "")
    if not neo4j_password:
        print("[graphiti] NEO4J_PASSWORD 未设置 → 使用本地内存模式（不写 Neo4j）")
        return
    print("[graphiti] 正在连接 Neo4j …")
    try:
        from graphiti_text_adventure.graphiti_adapter import connect_graphiti_from_env
        _graphiti_memory = run_async(connect_graphiti_from_env(), timeout=30)
        print("[graphiti] Neo4j 连接成功")

        # Create a separate driver for Cypher traversal
        from neo4j import AsyncGraphDatabase
        uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        user = os.environ.get("NEO4J_USER", "neo4j")
        _neo4j_driver = AsyncGraphDatabase.driver(uri, auth=(user, neo4j_password))
        _traversal = CypherTraversalProvider(
            query_runner_from_neo4j_driver(_neo4j_driver),
            protagonist_name="主角",
        )
        print("[graphiti] CypherTraversalProvider 就绪")

        if os.environ.get("SKIP_WORLD_INGEST") == "1":
            print("[graphiti] SKIP_WORLD_INGEST=1，跳过世界初始化写入")
        else:
            print("[graphiti] 开始写入世界数据（需调用多次 LLM，请稍候）…")
            from graphiti_text_adventure.ingestion import ingest_world
            run_async(ingest_world(_graphiti_memory, world, datetime(2024, 6, 1, 0, 0, 0)), timeout=600)
            print("[graphiti] 世界数据已写入 Neo4j 图谱 ✓")
    except Exception as exc:
        print(f"[graphiti] 初始化失败: {exc}")
        print("[graphiti] 降级为本地内存模式")
        _graphiti_memory = None
        _traversal = None
        _neo4j_driver = None


# ─── GameState ────────────────────────────────────────────────────────────────

class GameState:
    def __init__(self, world: World):
        self.world = world
        self.turn = 1
        self.day = 1
        self.time_index = 0
        self.location = world.locations[0].id
        self.selected_npc = world.locations[0].regulars[0] if world.locations[0].regulars else ""
        self.discovered: set[str] = set(world.protagonist.knows_secrets)
        self.known_facts: list[str] = self._initial_facts()
        self.lock = threading.Lock()

    def _initial_facts(self) -> list[str]:
        npc_by_id = {n.id: n for n in self.world.npcs}
        event_by_id = {e.id: e for e in self.world.public_events}
        facts = [self.world.protagonist.background]
        for link in self.world.protagonist.knows_npcs:
            npc = npc_by_id[link.npc_id]
            facts.append(f"认识{npc.full_name}（{npc.occupation}），{link.familiarity}。")
        for event_id in self.world.protagonist.knows_events:
            facts.append(event_by_id[event_id].description)
        return facts

    @property
    def time_label(self) -> str:
        return TIME_LABELS[self.time_index]

    def advance_time(self) -> None:
        self.turn += 1
        self.time_index = (self.time_index + 1) % len(TIME_LABELS)
        if self.time_index == 0:
            self.day += 1

    def maybe_discover(self, topic_id: str | None, reveal_level: str) -> list[str]:
        if not topic_id or reveal_level not in ("full", "hint"):
            return []
        secret_ids = {s.id for s in self.world.secrets}
        if topic_id in secret_ids and topic_id not in self.discovered:
            self.discovered.add(topic_id)
            secret = next(s for s in self.world.secrets if s.id == topic_id)
            self.known_facts.append(f"发现线索：{secret.description}")
            return [topic_id]
        return []

    def as_dict(self) -> dict:
        return {
            "turn": self.turn,
            "day": self.day,
            "time_label": self.time_label,
            "location": self.location,
            "selected_npc": self.selected_npc,
            "discovered_secrets": list(self.discovered),
            "clue_count": len(self.discovered),
            "known_facts": self.known_facts,
            "setting": {
                "town_name": self.world.setting.town_name,
                "era": self.world.setting.era,
                "atmosphere": self.world.setting.atmosphere,
            },
            "locations": {
                loc.id: {
                    "name": loc.name,
                    "description": loc.description,
                    "regulars": loc.regulars,
                }
                for loc in self.world.locations
            },
            "npcs": {
                npc.id: {
                    "name": npc.full_name,
                    "role": npc.occupation,
                    "mood": npc.current_state,
                }
                for npc in self.world.npcs
            },
            "secrets": {
                s.id: {
                    "title": _abbrev(s.description, 12),
                    "text": s.description,
                    "discovered": s.id in self.discovered,
                }
                for s in self.world.secrets
            },
            "public_events": [
                {"description": e.description, "when": e.when}
                for e in self.world.public_events
            ],
        }


def _abbrev(text: str, n: int = 10) -> str:
    return text[:n] + "…" if len(text) > n else text


# ─── In-memory context builders (fallback when no Graphiti) ──────────────────

def _build_player_view(state: GameState) -> list[MemoryEdge]:
    npc_by_id = {n.id: n for n in state.world.npcs}
    event_by_id = {e.id: e for e in state.world.public_events}
    secret_by_id = {s.id: s for s in state.world.secrets}
    facts: list[str] = []

    for link in state.world.protagonist.knows_npcs:
        npc = npc_by_id[link.npc_id]
        facts.append(
            f"主角认识{npc.full_name}（{npc.occupation}），"
            f"熟悉程度：{link.familiarity}，当前状态：{npc.current_state}"
        )
    for event_id in state.world.protagonist.knows_events:
        e = event_by_id[event_id]
        facts.append(f"公开事件：{e.description}（{e.when}）")
    for sid in state.discovered:
        if sid in secret_by_id:
            facts.append(f"主角已知线索：{secret_by_id[sid].description}")
    known_ids = {link.npc_id for link in state.world.protagonist.knows_npcs}
    for rel in state.world.npc_relationships:
        if rel.from_id in known_ids or rel.to_id in known_ids:
            facts.append(
                f"{npc_by_id[rel.from_id].full_name}对"
                f"{npc_by_id[rel.to_id].full_name}：{rel.type}（{rel.origin}）"
            )
    return [MemoryEdge(fact=f) for f in facts]


def _build_god_view(state: GameState, action: PlayerAction) -> list[TopicCandidate]:
    npc_by_id = {n.id: n for n in state.world.npcs}
    target_npc = next((n for n in state.world.npcs if n.full_name == (action.target or "")), None)
    current_loc = next((l for l in state.world.locations if l.id == state.location), None)
    loc_regulars = set(current_loc.regulars) if current_loc else set()

    scored: list[tuple[int, TopicCandidate]] = []
    for secret in state.world.secrets:
        if secret.id in state.discovered:
            continue
        score = {"高": 30, "中": 20, "低": 10}.get(secret.severity, 10)
        if target_npc and target_npc.id in secret.knowers:
            score += 50
        if loc_regulars & set(secret.knowers):
            score += 25
        knower_names = [npc_by_id[k].full_name for k in secret.knowers if k in npc_by_id]
        scored.append((score, TopicCandidate(
            topic_id=secret.id,
            name=_abbrev(secret.description),
            description=secret.description,
            topic_type="secret",
            truth_value=secret.truth_value,
            known_by_protagonist=False,
            last_seen_turn=None,
            source="world",
            knowers=knower_names,
        )))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [tc for _, tc in scored[:4]]


# ─── Context logging ──────────────────────────────────────────────────────────

def _log_context(ctx: DualContext) -> None:
    print("  ┌─ 主角视角")
    for edge in ctx.player_view:
        print(f"  │  · {edge.fact}")
    print(f"  └─ 上帝视角 Topics ({len(ctx.god_view_topics)}条)")
    for t in ctx.god_view_topics:
        knowers = "、".join(t.knowers) if t.knowers else "?"
        print(f"     [{t.topic_id}] {t.description}  (type={t.topic_type} knowers={knowers})")


# ─── Turn execution ───────────────────────────────────────────────────────────

async def _run_turn_async(
    state: GameState,
    narrator: OpenAICompatibleNarrator,
    verb: str,
    location: str,
    selected_npc: str,
) -> dict:
    with state.lock:
        state.location = location
        state.selected_npc = selected_npc

    npc_by_id = {n.id: n for n in state.world.npcs}
    loc_by_id = {l.id: l for l in state.world.locations}
    current_loc = loc_by_id.get(state.location)
    loc_name = current_loc.name if current_loc else ""
    present_names = [
        npc_by_id[nid].full_name
        for nid in (current_loc.regulars if current_loc else [])
        if nid in npc_by_id
    ]

    selected_obj = npc_by_id.get(selected_npc)
    if verb in ("打听", "观察", "帮忙") and selected_obj:
        target = selected_obj.full_name
    elif verb == "前往":
        target = loc_name
    else:
        target = None

    action = PlayerAction(verb=verb, target=target)

    ts = datetime.now().strftime("%H:%M:%S")
    label = f"{verb}{(' ' + target) if target else ''}"
    print(f"\n[{ts}] ══ 第{state.turn}回合 {label} ══ 地点={loc_name}")

    # Build context
    if _graphiti_memory is not None and _traversal is not None:
        print(f"[{ts}] 从 Graphiti 检索上下文 …")
        from graphiti_text_adventure.retrieval import build_player_view_query, build_semantic_topic_query, _dedupe
        ctx = await retrieve_dual_context(
            memory=_graphiti_memory,
            traversal=_traversal,
            protagonist_name="主角",
            location_name=loc_name,
            present_people=present_names,
            action=action,
            turn_number=state.turn,
        )
        print(f"[{ts}] Graphiti 检索完成")
    else:
        ctx = DualContext(
            player_view=_build_player_view(state),
            god_view_topics=_build_god_view(state, action),
        )
    _log_context(ctx)

    prompt = build_narration_prompt(
        location_name=loc_name,
        present_people=present_names,
        day=state.day,
        time_label=state.time_label,
        action=action,
        context=ctx,
    )

    # Narrate
    print(f"[{ts}] 调用 LLM …")
    try:
        raw = await narrator.narrate(prompt)
        resp = raw if isinstance(raw, dict) else json.loads(raw)
        decision = resp.get("decision", {})
        narrative = resp.get("narrative", "（叙述生成失败）")
        reveal_level = decision.get("reveal_level", "none")
        topic_id = decision.get("topic_id")
        reasoning = decision.get("reasoning", "")
        print(f"[{ts}] reveal={reveal_level}  topic={topic_id}")
        print(f"[{ts}] reasoning: {reasoning}")
        print(f"[{ts}] 叙述: {narrative}")
    except Exception as exc:
        print(f"[{ts}] 错误: {exc}")
        narrative = f"（LLM 请求失败：{exc}）"
        reveal_level, topic_id = "none", None

    # Store episodes to Graphiti if available
    if _graphiti_memory is not None:
        try:
            action_ep = EpisodeDraft(
                name=f"turn_{state.turn:03d}_action",
                body=json.dumps(action.to_dict(), ensure_ascii=False),
                source=EpisodeSource.JSON,
                source_description="player action",
            )
            narrative_ep = EpisodeDraft(
                name=f"turn_{state.turn:03d}_narrative",
                body=narrative,
                source=EpisodeSource.TEXT,
                source_description="narrative result",
            )
            await _graphiti_memory.add_episode(action_ep, _next_game_time())
            await _graphiti_memory.add_episode(narrative_ep, _next_game_time())
            print(f"[{ts}] episodes 已写入 Neo4j")
        except Exception as exc:
            print(f"[{ts}] episodes 写入失败: {exc}")

    with state.lock:
        newly = state.maybe_discover(topic_id, reveal_level)
        state.advance_time()
        state_dict = state.as_dict()

    if newly:
        print(f"[{ts}] 新揭露线索: {newly}")

    return {
        "narrative": narrative,
        "reveal_level": reveal_level,
        "topic_id": topic_id,
        "newly_discovered": newly,
        "state": state_dict,
    }


def run_turn(state, narrator, verb, location, selected_npc) -> dict:
    return run_async(_run_turn_async(state, narrator, verb, location, selected_npc))


# ─── HTTP Server ──────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    game_state: GameState
    narrator: OpenAICompatibleNarrator

    def log_message(self, fmt, *args) -> None:  # type: ignore[override]
        pass

    def _send_json(self, data: dict, status: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, ctype: str) -> None:
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in ("/", "/play.html"):
            self._send_file(WEB_ROOT / "play.html", "text/html; charset=utf-8")
        elif path == "/api/state":
            with self.game_state.lock:
                self._send_json(self.game_state.as_dict())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self) -> None:
        if self.path != "/api/action":
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        result = run_turn(
            state=self.game_state,
            narrator=self.narrator,
            verb=body.get("verb", "观察"),
            location=body.get("location", self.game_state.location),
            selected_npc=body.get("selected_npc", self.game_state.selected_npc),
        )
        self._send_json(result)


# ─── World summary logging ────────────────────────────────────────────────────

def _print_world_summary(world: World) -> None:
    npc_by_id = {n.id: n for n in world.npcs}
    print("=" * 60)
    print(f"  世界：{world.setting.town_name}（{world.setting.era}）")
    print(f"  {world.setting.atmosphere}")
    print("=" * 60)
    print(f"\nNPC ({len(world.npcs)}):")
    for n in world.npcs:
        tags = "、".join(n.personality_tags)
        print(f"  {n.id}  {n.full_name}  [{n.occupation}]  性格={tags}")
        print(f"         状态：{n.current_state}")
        print(f"         动机：{n.motivation}")

    print(f"\n地点 ({len(world.locations)}):")
    for loc in world.locations:
        regulars = "、".join(npc_by_id[r].full_name for r in loc.regulars if r in npc_by_id)
        print(f"  {loc.id}  {loc.name}  常驻={regulars}")
        print(f"         {loc.description}")

    print(f"\n公开事件 ({len(world.public_events)}):")
    for e in world.public_events:
        involved = "、".join(npc_by_id[r].full_name for r in e.involved_npcs if r in npc_by_id)
        print(f"  {e.id}  [{e.when}]  {e.description}  涉及={involved}")

    print(f"\n秘密 ({len(world.secrets)}):")
    for s in world.secrets:
        knowers = "、".join(npc_by_id[k].full_name for k in s.knowers if k in npc_by_id)
        print(f"  {s.id}  [{s.severity}] {s.description}")
        print(f"         知情者：{knowers}  真实性={s.truth_value}  关联={s.related_to}")

    print(f"\nNPC 关系 ({len(world.npc_relationships)}):")
    for r in world.npc_relationships:
        fn = npc_by_id[r.from_id].full_name
        tn = npc_by_id[r.to_id].full_name
        print(f"  {fn} → {tn}: {r.type}  ({r.origin})")

    print(f"\n主角：{world.protagonist.background}")
    for link in world.protagonist.knows_npcs:
        print(f"  认识 {npc_by_id[link.npc_id].full_name}（{link.familiarity}）")
    print(f"  已知事件: {world.protagonist.knows_events}")
    print()


# ─── Entry point ─────────────────────────────────────────────────────────────

def main() -> None:
    world_path = ROOT / "examples" / "stonebridge_world.json"
    with open(world_path, encoding="utf-8") as f:
        data = json.load(f)
    world = world_from_dict(data)
    _print_world_summary(world)

    state = GameState(world)
    narrator = OpenAICompatibleNarrator.from_env()

    # Optional Graphiti / Neo4j
    _init_graphiti(world)

    BoundHandler = type("BoundHandler", (Handler,), {
        "game_state": state,
        "narrator": narrator,
    })

    with ThreadingHTTPServer(("127.0.0.1", PORT), BoundHandler) as server:
        print(f"\n游戏服务器已启动：http://127.0.0.1:{PORT}/")
        print("Neo4j 模式：" + ("开启" if _graphiti_memory else "关闭（本地内存）"))
        print("按 Ctrl+C 退出\n")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\n服务器已停止")
        finally:
            if _neo4j_driver is not None:
                run_async(_neo4j_driver.close())


if __name__ == "__main__":
    main()
