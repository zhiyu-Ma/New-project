from __future__ import annotations

import http.server
import socketserver
from functools import partial
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = ROOT / "web"
PORT = 4173


class PlayableHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self) -> None:
        if self.path == "/":
            self.path = "/play.html"
        super().do_GET()


def main() -> None:
    handler = partial(PlayableHandler, directory=str(WEB_ROOT))
    with socketserver.TCPServer(("127.0.0.1", PORT), handler) as server:
        print(f"Playable page: http://127.0.0.1:{PORT}/play.html")
        server.serve_forever()


if __name__ == "__main__":
    main()
