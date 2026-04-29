from copy import deepcopy
from unittest import TestCase

from graphiti_text_adventure.models import EpisodeSource
from graphiti_text_adventure.world import (
    WorldValidationError,
    episodes_from_world,
    validate_world_data,
    world_from_dict,
)


def sample_world_data():
    return {
        "setting": {
            "town_name": "石桥镇",
            "era": "现代北方县城",
            "atmosphere": "熟人社会里每个人都知道一点别人的难处。",
        },
        "npcs": [
            {
                "id": "npc_01",
                "full_name": "张守义",
                "occupation": "铁匠铺老板",
                "personality_tags": ["嘴硬", "护短"],
                "current_state": "焦虑但强撑着照常开铺。",
                "motivation": "害怕债主找上女儿。",
            },
            {
                "id": "npc_02",
                "full_name": "李秋燕",
                "occupation": "小学校长",
                "personality_tags": ["谨慎", "重名声"],
                "current_state": "因学校账目被查而睡不好。",
                "motivation": "想保住学校。",
            },
            {
                "id": "npc_03",
                "full_name": "周明远",
                "occupation": "镇长",
                "personality_tags": ["圆滑", "爱面子"],
                "current_state": "忙着压下修桥款的流言。",
                "motivation": "害怕旧账曝光。",
            },
            {
                "id": "npc_04",
                "full_name": "何桂兰",
                "occupation": "杂货店老板",
                "personality_tags": ["精明", "多疑"],
                "current_state": "暗中记着每个人赊账的日子。",
                "motivation": "想拿回被拖欠的货款。",
            },
            {
                "id": "npc_05",
                "full_name": "陈启东",
                "occupation": "诊所医生",
                "personality_tags": ["温和", "回避冲突"],
                "current_state": "最近频繁夜里出诊。",
                "motivation": "害怕一年前的误诊被追究。",
            },
        ],
        "locations": [
            {
                "id": "loc_01",
                "name": "张守义铁匠铺",
                "description": "临街的小铺，炉火常年不断，柜台下压着账本。",
                "regulars": ["npc_01", "npc_04"],
            },
            {
                "id": "loc_02",
                "name": "石桥镇小学",
                "description": "旧教学楼刚粉刷过，办公室门总半掩着。",
                "regulars": ["npc_02", "npc_03"],
            },
            {
                "id": "loc_03",
                "name": "何桂兰杂货店",
                "description": "镇上消息最密的地方，货架后有一本厚赊账簿。",
                "regulars": ["npc_04", "npc_05"],
            },
        ],
        "public_events": [
            {
                "id": "event_01",
                "description": "石桥维修工程停了下来，镇上都说是材料款没批。",
                "involved_npcs": ["npc_03"],
                "when": "上个月",
            },
            {
                "id": "event_02",
                "description": "学校募捐箱少了一笔钱，李秋燕报了警。",
                "involved_npcs": ["npc_02", "npc_04"],
                "when": "上周",
            },
        ],
        "secrets": [
            {
                "id": "secret_01",
                "description": "张守义欠何桂兰一大笔高息货款。",
                "knowers": ["npc_01", "npc_04"],
                "truth_value": "true",
                "severity": "高",
                "related_to": ["secret_03"],
            },
            {
                "id": "secret_02",
                "description": "周明远把修桥款挪去补镇小学旧账。",
                "knowers": ["npc_02", "npc_03"],
                "truth_value": "半真半假",
                "severity": "高",
                "related_to": ["event_01", "secret_04"],
            },
            {
                "id": "secret_03",
                "description": "何桂兰保存着张守义抵押工具的字据。",
                "knowers": ["npc_04"],
                "truth_value": "true",
                "severity": "中",
                "related_to": ["secret_01"],
            },
            {
                "id": "secret_04",
                "description": "学校募捐款不是被偷，而是被临时借去付医药费。",
                "knowers": ["npc_02", "npc_05"],
                "truth_value": "true",
                "severity": "中",
                "related_to": ["event_02", "secret_02"],
            },
        ],
        "npc_relationships": [
            {
                "from": "npc_01",
                "to": "npc_04",
                "type": "distrust",
                "origin": "张守义觉得何桂兰逼债太狠。",
            },
            {
                "from": "npc_04",
                "to": "npc_01",
                "type": "distrust",
                "origin": "何桂兰认为张守义一直在拖延。",
            },
            {
                "from": "npc_02",
                "to": "npc_03",
                "type": "trust",
                "origin": "周明远曾帮学校争取过维修款。",
            },
            {
                "from": "npc_03",
                "to": "npc_02",
                "type": "trust",
                "origin": "李秋燕愿意替镇政府解释难处。",
            },
            {
                "from": "npc_05",
                "to": "npc_02",
                "type": "trust",
                "origin": "李秋燕帮陈启东照顾过病人家属。",
            },
            {
                "from": "npc_04",
                "to": "npc_05",
                "type": "distrust",
                "origin": "何桂兰怀疑陈启东替人隐瞒病情。",
            },
        ],
        "protagonist": {
            "background": "主角是返乡整理祖屋的本地青年。",
            "knows_npcs": [
                {"npc_id": "npc_01", "familiarity": "点头之交"},
                {"npc_id": "npc_04", "familiarity": "熟人"},
            ],
            "knows_events": ["event_01", "event_02"],
            "knows_secrets": [],
        },
    }


class WorldValidationTests(TestCase):
    def test_validates_required_story_topology(self):
        report = validate_world_data(sample_world_data())

        self.assertEqual(report.npc_count, 5)
        self.assertEqual(report.relationship_count, 6)
        self.assertEqual(report.secret_count, 4)
        self.assertEqual(report.multi_person_secret_count, 3)
        self.assertEqual(report.secret_event_link_count, 2)
        self.assertEqual(report.secret_cross_link_count, 4)

    def test_rejects_flat_secret_topology(self):
        data = deepcopy(sample_world_data())
        for secret in data["secrets"]:
            secret["knowers"] = [secret["knowers"][0]]
            secret["related_to"] = []

        with self.assertRaises(WorldValidationError) as error:
            validate_world_data(data)

        self.assertIn("at least 2 secrets involving multiple NPCs", error.exception.issues)
        self.assertIn("at least 1 secret linked to a public event", error.exception.issues)
        self.assertIn("at least 1 cross-link between secrets", error.exception.issues)


class EpisodeConversionTests(TestCase):
    def test_converts_world_json_into_compact_text_episodes(self):
        world = world_from_dict(sample_world_data())

        episodes = episodes_from_world(world)
        bodies = [episode.body for episode in episodes]

        self.assertEqual(25, len(episodes))
        self.assertTrue(all(episode.source == EpisodeSource.TEXT for episode in episodes))
        self.assertIn("张守义是铁匠铺老板。性格：嘴硬、护短。焦虑但强撑着照常开铺。想要或害怕：害怕债主找上女儿。", bodies)
        self.assertIn("张守义铁匠铺：临街的小铺，炉火常年不断，柜台下压着账本。常出现在这里的人：张守义、何桂兰。", bodies)
        self.assertIn("上个月，石桥维修工程停了下来，镇上都说是材料款没批。涉及人物：周明远。", bodies)
        self.assertIn("张守义、何桂兰知道一件事：张守义欠何桂兰一大笔高息货款。真实性：true。严重程度：高。关联：何桂兰保存着张守义抵押工具的字据。", bodies)
        self.assertIn("张守义对何桂兰的态度：distrust。原因：张守义觉得何桂兰逼债太狠。", bodies)
        self.assertIn("主角是返乡整理祖屋的本地青年。", bodies)
        self.assertIn("主角认识张守义，熟悉程度：点头之交。", bodies)
        self.assertIn("主角知道公开事件：学校募捐箱少了一笔钱，李秋燕报了警。", bodies)

    def test_unknown_ids_are_reported_before_episode_conversion(self):
        data = sample_world_data()
        data["locations"][0]["regulars"].append("npc_missing")

        with self.assertRaises(WorldValidationError) as error:
            world_from_dict(data)

        self.assertIn("unknown npc id npc_missing in location loc_01 regulars", error.exception.issues)

    def test_converts_initial_protagonist_known_secrets_into_episodes(self):
        data = sample_world_data()
        data["protagonist"]["knows_secrets"] = ["secret_01"]
        world = world_from_dict(data)

        episodes = episodes_from_world(world)
        bodies = [episode.body for episode in episodes]

        self.assertIn("主角知道秘密：张守义欠何桂兰一大笔高息货款。", bodies)
