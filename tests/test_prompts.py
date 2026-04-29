from unittest import TestCase

from graphiti_text_adventure.prompts import INITIALIZATION_PROMPT, build_narration_prompt
from graphiti_text_adventure.graphiti_adapter import MemoryEdge
from graphiti_text_adventure.models import PlayerAction
from graphiti_text_adventure.retrieval import DualContext, TopicCandidate


class PromptTests(TestCase):
    def test_initialization_prompt_contains_hard_story_constraints(self):
        self.assertIn("至少 2 个秘密涉及 2 个以上 NPC", INITIALIZATION_PROMPT)
        self.assertIn("严格按以下 JSON 输出", INITIALIZATION_PROMPT)
        self.assertIn('"topic_type"', INITIALIZATION_PROMPT)

    def test_narration_prompt_separates_player_view_from_hidden_fuel(self):
        context = DualContext(
            player_view=[MemoryEdge(uuid="edge-1", fact="主角认识张守义。")],
            god_view_topics=[
                TopicCandidate(
                    topic_id="secret_01",
                    name="高息货款",
                    description="张守义欠何桂兰一大笔高息货款。",
                    topic_type="secret",
                    truth_value="true",
                    known_by_protagonist=False,
                    last_seen_turn=None,
                    source="entity",
                    knowers=["张守义", "何桂兰"],
                )
            ],
        )

        prompt = build_narration_prompt(
            location_name="张守义铁匠铺",
            present_people=["张守义", "何桂兰"],
            day=1,
            time_label="上午",
            action=PlayerAction(verb="打听", target="张守义", modifier="财务状况"),
            context=context,
        )

        self.assertIn("# 主角的视角", prompt)
        self.assertIn("# 剧情燃料", prompt)
        self.assertIn("不允许的揭露方式", prompt)
        self.assertIn("[secret_01] 高息货款：张守义欠何桂兰一大笔高息货款。", prompt)
        self.assertIn("玩家动作\n打听-张守义-财务状况", prompt)
