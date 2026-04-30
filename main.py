"""
小镇模拟器 —— 基于 Graphiti 的自定义实体与边类型完整示例 (配置从 .env 读取)
运行前请确保：
1. 已创建 .env 文件并填入正确的密钥和连接信息
2. 已安装所需依赖: graphiti-core pydantic sentence-transformers python-dotenv
3. Neo4j 已启动并可连接
"""

import asyncio
import os
from datetime import datetime, timedelta
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from graphiti import Graphiti
from graphiti.types import EpisodeType
from graphiti.utils.search_filters import SearchFilters
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.embedder.sentence_transformers import SentenceTransformerEmbedder

# 加载 .env 文件中的环境变量
load_dotenv()

# =============================================================================
# 1. 自定义实体类型 (保持不变)
# =============================================================================
class Character(BaseModel):
    """A character in the town, including the player and NPCs."""
    full_name: str = Field(description="The character's full name.")
    age: Optional[int] = Field(default=None, description="Age if known.")
    occupation: Optional[str] = Field(default=None, description="Job or daily role.")
    personality: Optional[str] = Field(default=None, description="Brief personality traits.")
    is_player: bool = Field(default=False, description="True only for the player character.")

class Location(BaseModel):
    """A place in the town."""
    name: str = Field(description="Name of the location.")
    location_type: Optional[str] = Field(default=None, description="e.g., tavern, shop, house, plaza.")
    description: Optional[str] = Field(default=None, description="Short description of the place.")

class Item(BaseModel):
    """A significant item that can be owned or used."""
    name: str = Field(description="The item's name.")
    item_type: Optional[str] = Field(default=None, description="e.g., key, letter, weapon, heirloom.")

# =============================================================================
# 2. 自定义边类型 (保持不变)
# =============================================================================
class Knows(BaseModel):
    """Two characters know each other (basic acquaintance)."""

class Friends(BaseModel):
    """Two characters are friends."""

class Enemies(BaseModel):
    """Two characters are enemies or in conflict."""

class FamilyRelation(BaseModel):
    """A family tie between two characters."""
    relation_type: str = Field(..., description="e.g., parent, child, sibling, spouse.")

class Lovers(BaseModel):
    """Romantic involvement between characters."""

class ResidesAt(BaseModel):
    """A character lives at this location."""

class CurrentLocation(BaseModel):
    """Character is currently at this location (temporary, changes often)."""

class WorksAt(BaseModel):
    """The character works or runs a business at this location."""
    job_title: Optional[str] = Field(default=None, description="Optional job title.")

class Owns(BaseModel):
    """Character owns an item, building, or land."""
    ownership_detail: Optional[str] = Field(default=None, description="Optional detail.")

class InteractedWith(BaseModel):
    """Records a recent meaningful interaction between two characters."""
    interaction_type: Optional[str] = Field(default=None, description="e.g., conversation, trade, fight.")

# =============================================================================
# 3. 边类型映射 (保持不变)
# =============================================================================
EDGE_TYPE_MAP = {
    ("Character", "Character"): [
        "Knows", "Friends", "Enemies", "FamilyRelation",
        "Lovers", "InteractedWith"
    ],
    ("Character", "Location"): [
        "ResidesAt", "CurrentLocation", "WorksAt", "Owns"
    ],
    ("Character", "Item"): ["Owns"],
    ("Entity", "Entity"): []          # 泛化回退被禁用
}

ALL_EDGE_TYPES = {
    "Knows": Knows,
    "Friends": Friends,
    "Enemies": Enemies,
    "FamilyRelation": FamilyRelation,
    "Lovers": Lovers,
    "ResidesAt": ResidesAt,
    "CurrentLocation": CurrentLocation,
    "WorksAt": WorksAt,
    "Owns": Owns,
    "InteractedWith": InteractedWith,
}

EPISODE_CONFIG = {
    "entity_types": {
        "Character": Character,
        "Location": Location,
        "Item": Item
    },
    "edge_types": ALL_EDGE_TYPES,
    "edge_type_map": EDGE_TYPE_MAP,
    "excluded_entity_types": ["Entity"],
}

# =============================================================================
# 4. 从环境变量读取所有配置
# =============================================================================
# 大模型配置
LLM_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_BASE_URL = os.getenv("OPENAI_BASE_URL")
LLM_MODEL = os.getenv("OPENAI_MODEL")
LLM_SMALL_MODEL = os.getenv("OPENAI_SMALL_MODEL", LLM_MODEL)  # 若未单独设置，则复用主模型

# Neo4j 配置
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# 嵌入模型配置 (本地 Sentence Transformer)
EMBEDDER_MODEL_NAME = os.getenv("EMBEDDER_MODEL_NAME", "all-MiniLM-L6-v2")

# 验证必要配置
if not all([LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD]):
    raise ValueError(
        "缺少必要的环境变量，请检查 .env 文件是否包含 OPENAI_API_KEY, OPENAI_BASE_URL, "
        "OPENAI_MODEL, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD"
    )

# 构建 LLM 配置对象
llm_config = LLMConfig(
    api_key=LLM_API_KEY,
    base_url=LLM_BASE_URL,
    model=LLM_MODEL,
    small_model=LLM_SMALL_MODEL
)

# 构建本地嵌入器
embedder = SentenceTransformerEmbedder(model_name=EMBEDDER_MODEL_NAME)

# =============================================================================
# 5. 模拟 AI 生成故事 (实际项目中替换为真实 LLM 调用)
# =============================================================================
def generate_initial_story() -> str:
    return (
        "晨雾镇是一个宁静的西部小镇。主角名叫吉姆，是一名四处游历的枪手，刚刚抵达小镇。"
        "小镇上住着几位居民：铁匠老汤姆，他的儿子学徒小汤姆在铁匠铺帮忙；"
        "酒馆老板娘艾玛，她为人热情，是镇上消息最灵通的人；"
        "还有神秘的外来者杰克，最近才出现在镇外废弃的矿坑附近。"
        "吉姆在酒馆门口遇见了艾玛，两人聊了几句。"
    )

def generate_action_narrative(action: str, player_name: str, game_time: datetime) -> str:
    hour = game_time.strftime("%H:%M")
    base = f"{game_time.strftime('%Y年%m月%d日')} {hour}，"
    if "酒馆" in action:
        return base + f"{player_name}走进酒馆，向老板娘艾玛打听镇上的新鲜事。艾玛告诉他最近矿坑附近有奇怪的声音。"
    elif "铁匠铺" in action:
        return base + f"{player_name}来到铁匠铺，找老汤姆修理自己的左轮手枪。小汤姆在一旁专注地学习打铁。"
    elif "矿坑" in action:
        return base + f"{player_name}前往废弃矿坑，发现杰克正在鬼鬼祟祟地挖掘什么东西。两人对视了一眼，气氛有些紧张。"
    elif "对话" in action and "艾玛" in action:
        return base + f"{player_name}与艾玛在酒馆闲聊，艾玛提到杰克最近总是傍晚才回镇子。"
    else:
        return base + f"{player_name}在镇子上闲逛，观察着周围的人们。"

# =============================================================================
# 6. 游戏核心类 (封装 Graphiti 调用)
# =============================================================================
class TownSimulator:
    def __init__(self, graphiti: Graphiti, player_name: str, start_time: datetime):
        self.graphiti = graphiti
        self.player_name = player_name
        self.game_time = start_time
        self.action_index = 0

    async def initialize_world(self):
        story = generate_initial_story()
        print(f"[初始化] 生成小镇背景故事...")
        print(f"[初始化] 故事内容: {story[:80]}...")
        await self.add_game_episode(
            name="小镇诞生",
            body=story,
            source_type=EpisodeType.text,
            desc="小镇初始背景故事"
        )
        print("[初始化] 知识图谱已成功填充 ✅\n")

    async def player_action(self, action: str):
        self.game_time += timedelta(minutes=30)
        self.action_index += 1

        narrative = generate_action_narrative(action, self.player_name, self.game_time)
        print(f"[行动 {self.action_index}] {action}")
        print(f"  游戏时间: {self.game_time.strftime('%Y-%m-%d %H:%M')}")
        print(f"  叙述: {narrative[:100]}...")

        if "对话" in action:
            source_type = EpisodeType.message
            body = f"Player: {narrative}"
            desc = f"{self.player_name}的对话"
        else:
            source_type = EpisodeType.text
            body = narrative
            desc = f"{self.player_name}的行动: {action}"

        await self.add_game_episode(
            name=f"turn_{self.action_index:04d}_{action}_{self.game_time.strftime('%Y%m%d_%H%M')}",
            body=body,
            source_type=source_type,
            desc=desc
        )

    async def add_game_episode(self, name: str, body: str, source_type: EpisodeType, desc: str):
        await self.graphiti.add_episode(
            name=name,
            episode_body=body,
            source=source_type,
            source_description=desc,
            reference_time=self.game_time,
            **EPISODE_CONFIG
        )

    async def query_relationships(self, target_name: str):
        print(f"\n[查询] {self.player_name} 与 {target_name} 的关系：")
        results = await self.graphiti.search(
            query=f"{self.player_name} 和 {target_name} 之间的关系",
            center_node_entity_type="Character",
            filters=SearchFilters(edge_filters=["Friends", "Enemies", "Knows", "Lovers"]),
            num_results=5
        )
        if not results:
            print("  (暂无记录)")
        for edge in results:
            fact = edge.facts[0] if edge.facts else ""
            print(f"  - {edge.edge_type}: {fact}")
        print()

    async def query_current_location(self):
        print(f"\n[查询] {self.player_name} 当前所在地：")
        results = await self.graphiti.search(
            query=f"{self.player_name} 当前所在位置",
            center_node_entity_type="Character",
            filters=SearchFilters(edge_filters=["CurrentLocation"]),
            num_results=1
        )
        if results:
            fact = results[0].facts[0] if results[0].facts else ""
            print(f"  {fact}")
        else:
            print("  未知")
        print()

# =============================================================================
# 7. 创建 Graphiti 实例 (使用 .env 中的 Neo4j 凭证)
# =============================================================================
graphiti = Graphiti(
    uri=NEO4J_URI,
    user=NEO4J_USER,
    password=NEO4J_PASSWORD,
    llm_config=llm_config,
    embedder=embedder
)

# =============================================================================
# 8. 主函数：运行模拟
# =============================================================================
async def main():
    sim = TownSimulator(graphiti, "吉姆", datetime(1890, 4, 12, 8, 0))

    await sim.initialize_world()

    actions = [
        "前往酒馆",
        "与艾玛对话",
        "前往铁匠铺",
        "前往矿坑",
        "返回酒馆"
    ]

    for act in actions:
        await sim.player_action(act)
        await asyncio.sleep(0.5)

    await sim.query_relationships("艾玛")
    await sim.query_relationships("杰克")
    await sim.query_current_location()

    await graphiti.close()
    print("模拟结束，Graphiti 连接已关闭。")

if __name__ == "__main__":
    asyncio.run(main())
