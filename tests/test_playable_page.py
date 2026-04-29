from pathlib import Path
from unittest import TestCase


class PlayablePageTests(TestCase):
    def test_playable_page_contains_core_game_surface(self):
        html = Path("web/play.html").read_text(encoding="utf-8")

        self.assertIn("石桥镇", html)
        self.assertIn("data-action=", html)
        self.assertIn("storyLog", html)
        self.assertIn("worldState", html)

    def test_static_server_points_at_playable_page(self):
        server = Path("scripts/serve_playable.py").read_text(encoding="utf-8")

        self.assertIn("web", server)
        self.assertIn("play.html", server)
