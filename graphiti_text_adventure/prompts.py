from __future__ import annotations

from .models import PlayerAction
from .retrieval import DualContext, TopicCandidate


INITIALIZATION_PROMPT = """
# 任务

你要为一个文字冒险游戏生成一个小镇的初始世界状态。这个世界将作为知识图谱的种子数据，后续会随玩家行为演化。

# 设定

- 小镇规模：1 个小镇，约 5 个关键 NPC
- 风格：现实向，可以有戏剧冲突，但不要奇幻、超自然元素
- 时间锚点：故事开始前的过去一年内已发生若干事件

# 图谱 Schema

- Person：name, personality, current_mood
- Location：name, description
- Topic：事件/传闻/秘密/物品的统一抽象，属性包含 "topic_type" 和 "truth_value"

# 必须包含的元素

## NPC（5 个）
每人需要全名、职业/身份、2-3 个性格标签、当前状态、一个具体的想要或害怕的东西。

## 地点（3-5 个）
小镇内的关键场所，每个地点要标注谁常在此出没。

## 公开事件（2-3 个）
镇上人尽皆知的近期事件。这些事件要和至少一个 NPC 有牵连。

## 秘密（4-5 个）
每个秘密包含一句话描述、知情者列表、真实性、严重程度。

关键约束：
1. 至少 2 个秘密涉及 2 个以上 NPC
2. 至少 1 个秘密与某个公开事件相关
3. 至少 1 处信息不对称：A 知道关于 B 的某事，但 B 不知道 A 知道
4. 秘密之间至少有 1 处交叉（一个秘密的揭露会牵出另一个秘密）

## NPC 之间的关系（至少 6 条）
关系类型例如：信任、敌意、欠人情、暗恋、生意伙伴、远亲等。

## 主角与世界的初始连接
主角可以是外来者、本地人或返乡者；主角可以认识部分 NPC，建议初始化时不知道任何秘密。

# 输出格式

严格按以下 JSON 输出，不要额外说明：

{
  "setting": {"town_name": "...", "era": "...", "atmosphere": "..."},
  "npcs": [{"id": "npc_01", "full_name": "...", "occupation": "...", "personality_tags": ["...", "..."], "current_state": "...", "motivation": "..."}],
  "locations": [{"id": "loc_01", "name": "...", "description": "...", "regulars": ["npc_01"]}],
  "public_events": [{"id": "event_01", "description": "...", "involved_npcs": ["npc_02"], "when": "上周"}],
  "secrets": [{"id": "secret_01", "description": "...", "knowers": ["npc_01"], "truth_value": "true", "severity": "...", "related_to": ["event_01"]}],
  "npc_relationships": [{"from": "npc_01", "to": "npc_02", "type": "distrust", "origin": "..."}],
  "protagonist": {"background": "...", "knows_npcs": [{"npc_id": "npc_01", "familiarity": "点头之交"}], "knows_events": ["event_01"], "knows_secrets": []}
}
""".strip()


def build_narration_prompt(
    location_name: str,
    present_people: list[str],
    day: int,
    time_label: str,
    action: PlayerAction,
    context: DualContext,
) -> str:
    player_view = "\n".join(f"- {edge.fact}" for edge in context.player_view) or "- 无明确已知信息"
    hidden_topics = "\n".join(_format_topic(topic) for topic in context.god_view_topics) or "- 无潜在隐藏信息"
    people = "、".join(present_people) if present_people else "无"
    return f"""
# 角色
你是一个文字冒险游戏的叙述者，负责生成自然、真实的事件文本。

# 当前场景
地点：{location_name}
在场人物：{people}
游戏时间：第 {day} 天 {time_label}

# 主角的视角
这些信息主角已知，可在叙述中自然体现：
{player_view}

# 剧情燃料
这些是隐藏信息，主角不知道，不要直接用旁白曝光：
{hidden_topics}

# 玩家动作
{action.to_menu_label()}

# 你的任务

第一步：判断本回合是否揭露任何隐藏信息。
- 揭露条件：主角的动作与某 Topic 高度相关，且场景合理
- 揭露程度：full / hint / trace / none
- 多数情况下应选 hint、trace 或 none，避免剧情过快推进

第二步：生成 100-200 字叙述。
- 写真实的对话和行为，不写说明文
- 如果选择揭露，让信息通过 NPC 的反应、环境细节、口误等方式自然透出
- 如果选择不涉及，写出真实的无进展感

# 揭露的实现通道

通道 1：NPC 自己的反应失控
通道 2：环境物证
通道 3：第三方在场
通道 4：主角的推理（必须基于主角视角已知信息）

不允许的揭露方式：
- NPC 主动倾诉秘密（除非有强烈剧情理由）
- 旁白上帝视角直接说明
- 主角的直觉（除非已知信息支撑）

# 命名一致性
叙述中所有 NPC 必须使用其全名，不使用别名、绰号或职业称谓。对话中 NPC 称呼可以自然，但叙述部分必须用全名。

# 输出格式

{{
  "decision": {{
    "topic_id": "Topic-A" | null,
    "reveal_level": "full" | "hint" | "trace" | "none",
    "reasoning": "一句话说明为什么这样处理"
  }},
  "narrative": "100-200 字的叙述文本"
}}
""".strip()


def _format_topic(topic: TopicCandidate) -> str:
    knowers = "、".join(topic.knowers) if topic.knowers else "未知"
    last_seen = "从未" if topic.last_seen_turn is None else f"第 {topic.last_seen_turn} 回合"
    return (
        f"[{topic.topic_id}] {topic.name}：{topic.description}\n"
        f"  - 类型：{topic.topic_type}\n"
        f"  - 真实性：{topic.truth_value}\n"
        f"  - 知情者：{knowers}\n"
        f"  - 主角已知：{'是' if topic.known_by_protagonist else '否'}\n"
        f"  - 上次在叙述中出现：{last_seen}"
    )
