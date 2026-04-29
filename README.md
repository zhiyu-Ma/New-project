# Graphiti 文字冒险 Demo

基于 Graphiti 知识图谱的文字冒险游戏 Demo，同时验证知识图谱在动态叙事场景中的能力。

## 快速开始（可玩版）

### 环境要求

- Python 3.10+
- `.env` 文件中填好 `OPENAI_API_KEY`（支持 OpenAI 兼容接口）

### 第一次启动

```bash
cd "New project"
python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[real]'
cp .env.example .env
# 编辑 .env，填入 OPENAI_API_KEY（和可选的 OPENAI_BASE_URL、OPENAI_MODEL）
```

### 启动游戏

```bash
. .venv/bin/activate
python scripts/game_server.py
```

然后打开 **http://127.0.0.1:4173/**

### `.env` 最少配置

```bash
OPENAI_API_KEY=你的密钥
# 如果用代理或非 OpenAI 模型：
# OPENAI_BASE_URL=https://your-proxy.com/v1
# OPENAI_MODEL=gpt-4.1-mini
```

---

## 玩法说明

| 操作 | 说明 |
|------|------|
| 点击地点 | 前往该地点，AI 叙述到达场景 |
| 点击 NPC 卡片 | 选中交互对象 |
| **打听** | 向选中 NPC 打听消息 |
| **观察** | 观察选中 NPC 或周围环境 |
| **帮忙** | 协助选中 NPC 做事 |
| **等一会** | 时间流逝，什么都不做 |

右侧面板显示"主角已知"和"隐藏线索"。线索揭露后会高亮显示并进入已知信息。

---

## 架构

### 关键路径（每回合）

```
玩家动作 (verb + location + selected_npc)
  → game_server.py：从世界数据构建双重上下文
    ├─ 主角视角（已知 NPC、公开事件、已揭露线索、NPC 关系）
    └─ 上帝视角（未揭露秘密，按相关度排序）
  → build_narration_prompt() → LLM（OpenAI 兼容接口）
  → 解析 decision.reveal_level / topic_id
  → 更新 GameState（揭露秘密、时间推进）
  → 返回 JSON 给前端
```

### 模块说明

| 模块 | 作用 |
|------|------|
| `scripts/game_server.py` | HTTP 服务器，游戏主逻辑 |
| `web/play.html` | 浏览器前端，调用 `/api/action` |
| `graphiti_text_adventure/world.py` | `World` dataclass、世界约束校验、episode 生成 |
| `graphiti_text_adventure/prompts.py` | 构造叙述 prompt |
| `graphiti_text_adventure/narrator.py` | OpenAI 兼容叙述器 |
| `graphiti_text_adventure/retrieval.py` | 双上下文检索（含 Graphiti/Neo4j 路径） |
| `graphiti_text_adventure/graphiti_adapter.py` | Graphiti 适配层 |
| `graphiti_text_adventure/engine.py` | 单回合编排（供 Graphiti 路径使用） |
| `examples/stonebridge_world.json` | 默认世界：石桥镇 |

---

## 带 Graphiti/Neo4j 的完整模式

`game_server.py` 目前使用世界 JSON 直接构建上下文（无需 Neo4j），适合直接试玩。

如需启用完整 Graphiti 知识图谱（世界初始化 + 运行时图谱更新），还需要：

1. 启动 Neo4j（本地或远端），确保 `.env` 中 `NEO4J_URI`、`NEO4J_USER`、`NEO4J_PASSWORD` 正确
2. 运行 smoke test 验证连接：
   ```bash
   . .venv/bin/activate
   python scripts/real_graphiti_smoke.py
   ```

---

## 开发工具

```bash
# 打印世界生成 prompt（供 LLM 生成新世界 JSON 用）
python3 -m graphiti_text_adventure.cli init-prompt

# 校验世界文件
python3 -m graphiti_text_adventure.cli validate-world examples/stonebridge_world.json

# 打印初始化 episodes（JSONL）
python3 -m graphiti_text_adventure.cli episodes examples/stonebridge_world.json

# 单元测试
python3 -m unittest discover -s tests -v

# 全量检查
scripts/check.sh
```

---

## 世界约束

`validate_world_data()` 强制要求：

- 恰好 5 个 NPC
- 至少 6 条 NPC 关系
- 至少 4 个秘密，其中至少 2 个涉及多个 NPC
- 至少 1 个秘密关联公开事件
- 至少 1 个秘密与另一个秘密交叉关联
- 所有 `id` 交叉引用可解析
