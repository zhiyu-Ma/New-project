from unittest import TestCase

from graphiti_text_adventure.graphiti_adapter import MemoryEdge
from graphiti_text_adventure.models import PlayerAction
from graphiti_text_adventure.retrieval import (
    CypherTraversalProvider,
    DualContext,
    GraphTraversalProvider,
    TopicCandidate,
    build_player_view_query,
    build_semantic_topic_query,
    rank_topics,
    retrieve_dual_context,
    query_runner_from_neo4j_driver,
)


class FakeTraversal(GraphTraversalProvider):
    def __init__(self):
        self.entity_topics = []
        self.location_topics = []

    async def topics_near_entities(self, entity_names, hops=2):
        self.entity_request = (tuple(entity_names), hops)
        return self.entity_topics

    async def topics_near_location(self, location_name):
        self.location_request = location_name
        return self.location_topics


class FakeMemory:
    def __init__(self):
        self.responses = [
            [MemoryEdge(uuid="edge-1", fact="主角认识张守义。")],
            [],
        ]
        self.search_requests = []

    async def search(self, query, **kwargs):
        self.search_requests.append((query, kwargs))
        self.search_request = (query, kwargs)
        return self.responses.pop(0)


class RetrievalTests(TestCase):
    def test_builds_player_view_query_from_current_scene(self):
        action = PlayerAction(verb="打听", target="张守义", modifier="财务状况")

        query = build_player_view_query(
            protagonist_name="主角",
            location_name="张守义铁匠铺",
            present_people=["张守义", "何桂兰"],
            action=action,
        )

        self.assertEqual(query, "主角在张守义铁匠铺对张守义执行打听，话题或修饰为财务状况。检索主角已知的人、地点、事件、传闻、秘密和物品。")

    def test_builds_semantic_topic_query_from_action(self):
        action = PlayerAction(verb="赠送", target="李秋燕", modifier="药费收据")

        self.assertEqual(
            build_semantic_topic_query("石桥镇小学", ["李秋燕"], action),
            "石桥镇小学；在场人物：李秋燕；玩家动作：赠送 李秋燕 药费收据；检索潜在相关的事件、秘密、传闻、物品 Topic。",
        )

    def test_ranks_hidden_secret_topics_before_known_or_stale_events(self):
        candidates = [
            TopicCandidate(
                topic_id="event_01",
                name="修桥停工",
                description="公开事件",
                topic_type="event",
                truth_value="true",
                known_by_protagonist=True,
                last_seen_turn=9,
                source="semantic",
            ),
            TopicCandidate(
                topic_id="secret_01",
                name="高息货款",
                description="张守义欠何桂兰钱",
                topic_type="secret",
                truth_value="true",
                known_by_protagonist=False,
                last_seen_turn=None,
                source="entity",
            ),
            TopicCandidate(
                topic_id="rumor_01",
                name="夜间出诊",
                description="陈启东夜里出门",
                topic_type="rumor",
                truth_value="unknown",
                known_by_protagonist=False,
                last_seen_turn=2,
                source="location",
            ),
        ]

        ranked = rank_topics(candidates, current_turn=10, limit=2)

        self.assertEqual([topic.topic_id for topic in ranked], ["secret_01", "rumor_01"])

    async def _retrieve(self):
        memory = FakeMemory()
        traversal = FakeTraversal()
        traversal.entity_topics = [
            TopicCandidate("secret_01", "高息货款", "张守义欠何桂兰钱", "secret", "true", False, None, "entity")
        ]
        traversal.location_topics = [
            TopicCandidate("event_01", "修桥停工", "公开事件", "event", "true", True, 1, "location")
        ]
        action = PlayerAction(verb="打听", target="张守义", modifier="财务状况")

        context = await retrieve_dual_context(
            memory=memory,
            traversal=traversal,
            protagonist_name="主角",
            location_name="张守义铁匠铺",
            present_people=["张守义", "何桂兰"],
            action=action,
            turn_number=4,
        )

        return context, memory, traversal

    def test_retrieve_dual_context_combines_player_and_god_views(self):
        import asyncio

        context, memory, traversal = asyncio.run(self._retrieve())

        self.assertIsInstance(context, DualContext)
        self.assertEqual([edge.fact for edge in context.player_view], ["主角认识张守义。"])
        self.assertEqual([topic.topic_id for topic in context.god_view_topics], ["secret_01", "event_01"])
        self.assertEqual(traversal.entity_request, (("张守义", "何桂兰", "张守义铁匠铺"), 2))
        self.assertEqual(traversal.location_request, "张守义铁匠铺")
        self.assertIn("玩家动作：打听 张守义 财务状况", memory.search_request[0])

    def test_retrieve_dual_context_includes_semantic_search_topics(self):
        import asyncio

        memory = FakeMemory()
        memory.responses = [
            [MemoryEdge(uuid="edge-1", fact="主角认识张守义。")],
            [
                MemoryEdge(
                    uuid="semantic-edge-1",
                    fact="传闻：何桂兰保存着张守义抵押工具的字据。",
                    attributes={"topic_id": "secret_03", "topic_type": "secret", "truth_value": "true"},
                )
            ],
        ]
        traversal = FakeTraversal()

        context = asyncio.run(
            retrieve_dual_context(
                memory=memory,
                traversal=traversal,
                protagonist_name="主角",
                location_name="张守义铁匠铺",
                present_people=["张守义"],
                action=PlayerAction(verb="打听", target="张守义", modifier="抵押工具"),
                turn_number=3,
            )
        )

        self.assertEqual([topic.topic_id for topic in context.god_view_topics], ["secret_03"])
        self.assertEqual(context.god_view_topics[0].source, "semantic")
        self.assertEqual(context.god_view_topics[0].description, "传闻：何桂兰保存着张守义抵押工具的字据。")


class CypherTraversalProviderTests(TestCase):
    def test_maps_entity_traversal_rows_into_topic_candidates(self):
        calls = []

        async def query_runner(query, params):
            calls.append((query, params))
            return [
                {
                    "topic_id": "secret_01",
                    "name": "高息货款",
                    "description": "张守义欠何桂兰钱",
                    "topic_type": "secret",
                    "truth_value": "true",
                    "known_by_protagonist": False,
                    "last_seen_turn": None,
                    "knowers": ["张守义", "何桂兰"],
                }
            ]

        import asyncio

        provider = CypherTraversalProvider(query_runner)
        topics = asyncio.run(provider.topics_near_entities(["张守义"], hops=2))

        self.assertEqual(topics[0].topic_id, "secret_01")
        self.assertEqual(topics[0].source, "entity")
        self.assertEqual(topics[0].knowers, ["张守义", "何桂兰"])
        self.assertIn("[*1..2]", calls[0][0])
        self.assertEqual(calls[0][1], {"entity_names": ["张守义"], "protagonist_name": "主角"})

    def test_maps_location_rows_into_topic_candidates(self):
        calls = []

        async def query_runner(query, params):
            calls.append((query, params))
            return [
                {
                    "topic_id": "event_01",
                    "name": "修桥停工",
                    "description": "修桥工程停了",
                    "topic_type": "event",
                    "truth_value": "true",
                    "known_by_protagonist": True,
                    "last_seen_turn": 1,
                    "knowers": [],
                }
            ]

        import asyncio

        provider = CypherTraversalProvider(query_runner)
        topics = asyncio.run(provider.topics_near_location("张守义铁匠铺"))

        self.assertEqual(topics[0].topic_id, "event_01")
        self.assertEqual(topics[0].source, "location")
        self.assertIn("location.name = $location_name", calls[0][0])
        self.assertEqual(calls[0][1], {"location_name": "张守义铁匠铺", "protagonist_name": "主角"})

    def test_builds_query_runner_from_neo4j_style_driver(self):
        class FakeDriver:
            async def execute_query(self, query, **params):
                self.call = (query, params)
                return ([{"name": "高息货款"}], None, None)

        import asyncio

        driver = FakeDriver()
        runner = query_runner_from_neo4j_driver(driver)
        rows = asyncio.run(runner("MATCH (n) RETURN n", {"limit": 1}))

        self.assertEqual(rows, [{"name": "高息货款"}])
        self.assertEqual(driver.call, ("MATCH (n) RETURN n", {"limit": 1}))
