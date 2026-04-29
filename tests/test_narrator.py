import asyncio
import json
from unittest import TestCase

from graphiti_text_adventure.narrator import OpenAICompatibleNarrator, parse_json_object


class NarratorTests(TestCase):
    def test_parse_json_object_accepts_plain_json_or_fenced_json(self):
        expected = {
            "decision": {"topic_id": None, "reveal_level": "none", "reasoning": "无"},
            "narrative": "张守义没有接话。",
        }

        self.assertEqual(parse_json_object(json.dumps(expected, ensure_ascii=False)), expected)
        self.assertEqual(parse_json_object("```json\n" + json.dumps(expected, ensure_ascii=False) + "\n```"), expected)

    def test_openai_compatible_narrator_sends_prompt_and_parses_message_content(self):
        calls = []

        async def transport(url, headers, payload):
            calls.append((url, headers, payload))
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "decision": {
                                        "topic_id": "secret_01",
                                        "reveal_level": "hint",
                                        "reasoning": "动作相关。",
                                    },
                                    "narrative": "张守义把字条往柜台里推。",
                                },
                                ensure_ascii=False,
                            )
                        }
                    }
                ]
            }

        narrator = OpenAICompatibleNarrator(
            api_key="test-key",
            model="test-model",
            base_url="https://example.test/v1",
            transport=transport,
        )

        result = asyncio.run(narrator.narrate("prompt text"))

        self.assertEqual(result["decision"]["topic_id"], "secret_01")
        self.assertEqual(calls[0][0], "https://example.test/v1/chat/completions")
        self.assertEqual(calls[0][1]["Authorization"], "Bearer test-key")
        self.assertEqual(calls[0][2]["model"], "test-model")
        self.assertEqual(calls[0][2]["messages"][0]["content"], "prompt text")
