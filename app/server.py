"""Small HTTP server that publishes the overlay.

Point an OBS browser source at http://127.0.0.1:8777/. The page pulls only the
standings, so rows can move smoothly instead of the whole page reloading.



    server = OverlayServer(port=8777)
    server.start()
    server.update(standings, round_no=2)
"""

from __future__ import annotations

PROTOCOL = 2  # module interface version; mixing different values is unsafe
FILE_SET = "2026-09-04-k"  # release this file belongs to

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from overlay import live_page
from settings import OVERLAY_PORT, OVERLAY_PORT_TRIES, OVERLAY_TITLE


class OverlayServer:
    """Hold the current standings and serve them on request."""

    def __init__(self, port: int = OVERLAY_PORT, title: str = OVERLAY_TITLE) -> None:
        self.port = port
        self.title = title
        self._state: dict = {"title": title, "rows": [], "version": 0}
        self._lock = threading.Lock()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    # ---- state ---------------------------------------------------------
    def update(
        self,
        standings,
        round_no: int = 0,
        out_slots: set[int] | None = None,
    ) -> None:
        """Store the current standings.

        Args:
            standings: ordered TeamStanding list.
            round_no: number of finished rounds.
            out_slots: slots already eliminated in the current round.
        """
        out_slots = out_slots or set()
        rows = [
            {
                "slot": team.slot,
                "name": team.name,
                "total": round(float(team.total), 1),
                "ks": round(float(team.kill_score), 1),
                "place": team.place_score,
                "penalty": round(float(team.penalty), 1),
                "out": team.slot in out_slots,
            }
            for team in standings
        ]
        with self._lock:
            self._state = {
                "title": self.title,
                "rows": rows,
                "round": round_no,
                "version": self._state["version"] + 1,
            }

    def state_json(self) -> bytes:
        with self._lock:
            return json.dumps(self._state, ensure_ascii=False).encode("utf-8")

    # ---- server --------------------------------------------------------
    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/"

    def start(self, tries: int = OVERLAY_PORT_TRIES) -> None:
        """Start the server, moving to the next free port if needed.

        The port actually opened is available as self.port and self.url.

        Args:
            tries: how many consecutive ports to try.

        Raises:
            OSError: every port in the range is taken.
        """
        if self._server is not None:
            return
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:  # keep the console quiet
                pass

            def _send(self, body: bytes, kind: str) -> None:
                self.send_response(200)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                if self.path.startswith("/state.json"):
                    self._send(server.state_json(), "application/json; charset=utf-8")
                elif self.path in ("/", "/index.html", "/overlay"):
                    self._send(
                        live_page(server.title).encode("utf-8"), "text/html; charset=utf-8"
                    )
                else:
                    self.send_error(404)

        first = self.port
        for step in range(tries):
            try:
                self._server = ThreadingHTTPServer(("127.0.0.1", self.port), Handler)
                break
            except OSError:
                self.port += 1
        else:
            raise OSError(
                f"{first}번부터 {first + tries - 1}번까지 모든 포트가 사용 중입니다. "
                "config.json의 overlay_port를 다른 값으로 바꿔 주세요."
            )
        if self.port != first:
            print(f"{first}번 포트가 사용 중이어서 {self.port}번으로 열었습니다")

        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
