"""Capture a single window.

The target is given either by process name, which survives title changes, or
by part of the title. Neither is stored in the config file; it is supplied on
each run.

Windows Graphics Capture is used when available; otherwise the visible screen
region of the window is grabbed instead.
"""

from __future__ import annotations

PROTOCOL = 2  # module interface version; mixing different values is unsafe
FILE_SET = "2026-09-06-a"  # release this file belongs to

import threading
import time
from dataclasses import dataclass

import numpy as np


@dataclass
class WindowInfo:
    hwnd: int
    title: str
    process: str


def _process_of(hwnd: int) -> str:
    import win32api
    import win32con
    import win32process

    try:
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        handle = win32api.OpenProcess(
            win32con.PROCESS_QUERY_LIMITED_INFORMATION, False, pid
        )
        try:
            path = win32process.GetModuleFileNameEx(handle, 0)
        finally:
            win32api.CloseHandle(handle)
        return path.rsplit("\\", 1)[-1]
    except Exception:
        return ""


def list_windows() -> list[WindowInfo]:
    """Return the visible windows that have a title."""
    import win32gui

    found: list[WindowInfo] = []

    def collect(hwnd, _):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if title:
                found.append(WindowInfo(hwnd, title, _process_of(hwnd)))

    win32gui.EnumWindows(collect, None)
    return found


def find_window(
    process: str | None = None,
    title: str | None = None,
    hwnd: int | None = None,
) -> WindowInfo:
    """Find the target window.

    A window handle takes priority: several Chrome windows share one process and
    similar titles, so a name lookup would always return the first one. The
    handle is used while it lives; after that the name is used again.

    Args:
        process: process name, e.g. "chrome.exe".
        title: part of the window title.
        hwnd: handle of the window chosen earlier.

    Returns:
        The matching WindowInfo.

    Raises:
        RuntimeError: no window matches.
    """
    windows = list_windows()
    if hwnd is not None:
        for w in windows:
            if w.hwnd == hwnd:
                return w
    if title:
        exact = [w for w in windows if w.title == title]
        if exact:
            return exact[0]
    if process:
        want = process.lower()
        hits = [w for w in windows if w.process.lower() == want]
        if not hits:
            hits = [w for w in windows if want in w.process.lower()]
        if hits:
            return max(hits, key=lambda w: len(w.title))
    if title:
        hits = [w for w in windows if title.lower() in w.title.lower()]
        if hits:
            return hits[0]
    raise RuntimeError(f"창을 찾을 수 없습니다 (프로세스 {process}, 제목 {title})")


def pick_target() -> tuple[str | None, str | None]:
    """Print the window list and return the chosen (process, title)."""
    windows = list_windows()
    for i, w in enumerate(windows):
        print(f"{i:3d}  {w.process:24s}  {w.title}")
    chosen = windows[int(input("번호 입력: "))]
    print(f"프로세스 {chosen.process} 기준으로 추적합니다. 창 제목이 바뀌어도 유지됩니다.")
    return (chosen.process or None), chosen.title


class WindowSource:
    """Serve the latest frame of one window as an RGB array.

    If the title changes or the window is recreated, the process name is used to
    reattach, so a scene change during a broadcast does not stop the reader.
    """

    def __init__(
        self,
        process: str | None = None,
        title: str | None = None,
        hwnd: int | None = None,
        prefer_wgc: bool = True,
        retry_after: float = 3.0,
    ) -> None:
        self.process = process
        self.title = title
        self.hwnd = hwnd  # chosen window; used when several share a name
        self.prefer_wgc = prefer_wgc
        self.retry_after = retry_after

        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._control = None
        self._mode = "none"
        self._last_ok = time.time()
        self.last_reason = ""  # why the last grab failed
        self._wgc_since = 0.0  # when the WGC session started
        self._attach()

    def _attach(self) -> None:
        target = find_window(self.process, self.title, self.hwnd)
        self.title = target.title
        self.hwnd = target.hwnd
        self._hwnd = target.hwnd
        self._last_ok = time.time()
        if getattr(self, "_forced_screen", False):
            self._start_screen()
            self._mode = "screen"
        elif self.prefer_wgc and self._start_wgc(target.title):
            self._mode = "wgc"
            self._wgc_since = time.time()
        else:
            self._start_screen()
            self._mode = "screen"

    def _start_wgc(self, title: str) -> bool:
        try:
            from windows_capture import WindowsCapture
        except ImportError:
            return False
        try:
            capture = WindowsCapture(
                cursor_capture=False, draw_border=False, window_name=title
            )
        except Exception:
            return False

        @capture.event
        def on_frame_arrived(frame, capture_control):  # pragma: no cover - Windows only
            bgra = frame.frame_buffer
            with self._lock:
                self._frame = bgra[:, :, [2, 1, 0]].copy()

        @capture.event
        def on_closed():  # pragma: no cover - Windows only
            with self._lock:
                self._frame = None

        self._stop_wgc()
        self._control = capture.start_free_threaded()
        return True

    def _stop_wgc(self) -> None:
        if self._control is not None:
            try:
                self._control.stop()
            except Exception:
                pass
            self._control = None

    def _start_screen(self) -> None:
        import mss

        self._sct = mss.mss()

    WGC_WAIT = 1.0  # seconds to wait for WGC before falling back

    def _fall_back_to_screen(self) -> None:
        """Fall back to screen grabbing when WGC delivers nothing.

        Some windows, browsers in particular, never deliver a WGC frame even
        when the right window was picked. Waiting forever would read nothing, so
        the visible screen region is grabbed instead.
        """
        self._stop_wgc()
        self._forced_screen = True  # stay on screen grabbing after a fallback
        self._start_screen()
        self._mode = "screen"
        print(f"'{self.title}' 창에서 WGC 프레임이 오지 않아 화면 캡처로 전환합니다")

    def grab(self) -> np.ndarray | None:
        frame = self._grab_once()
        if frame is None and self._mode == "wgc":
            if time.time() - self._wgc_since >= self.WGC_WAIT:
                self._fall_back_to_screen()
                frame = self._grab_once()
        if frame is not None:
            self._last_ok = time.time()
            self.last_reason = ""
            return frame
        if time.time() - self._last_ok >= self.retry_after:
            try:
                self._attach()
            except RuntimeError as e:
                self.last_reason = (
                    f"{e}. 창이 닫혔거나 제목이 변경되었습니다. "
                    "--process 또는 --window로 대상을 다시 지정해 주세요"
                )
                return None
            return self._grab_once()
        return None

    def _grab_once(self) -> np.ndarray | None:
        """Return one frame, or None with the reason recorded in last_reason."""
        if self._mode == "wgc":
            with self._lock:
                if self._frame is None:
                    self.last_reason = (
                        f"'{self.title}' 창에서 새 프레임이 오지 않았습니다. "
                        "창이 최소화되었거나 화면이 정지한 경우 프레임이 오지 않습니다"
                    )
                    return None
                return self._frame.copy()

        import win32gui

        try:
            left, top, right, bottom = win32gui.GetClientRect(self._hwnd)
            left, top = win32gui.ClientToScreen(self._hwnd, (left, top))
            right, bottom = win32gui.ClientToScreen(self._hwnd, (right, bottom))
        except Exception as e:
            self.last_reason = f"'{self.title}' 창의 좌표를 얻지 못했습니다({e}). 창이 닫힌 것으로 보입니다"
            return None
        box = {"left": left, "top": top, "width": right - left, "height": bottom - top}
        if box["width"] <= 0 or box["height"] <= 0:
            self.last_reason = (
                f"'{self.title}' 창의 크기가 0입니다. 최소화 상태에서는 캡처할 수 없습니다"
            )
            return None
        return np.array(self._sct.grab(box))[:, :, [2, 1, 0]]

    def first_frame(self, timeout: float = 5.0) -> np.ndarray:
        """Wait for the first frame.

        Args:
            timeout: seconds to wait before giving up.

        Returns:
            The first frame as an RGB array.

        Raises:
            SystemExit: no frame arrived in time.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            frame = self.grab()
            if frame is not None:
                return frame
            time.sleep(0.1)
        raise SystemExit(f"창 프레임을 받지 못했습니다. {self.last_reason}")

    def close(self) -> None:
        self._stop_wgc()


def add_target_args(parser) -> None:
    """Add the shared target options to an argument parser."""
    parser.add_argument("--process", help="대상 프로세스 name (예: EternalReturn.exe)")
    parser.add_argument("--window", help="대상 창 제목의 일부")
    parser.add_argument(
        "--capture",
        choices=("auto", "wgc", "screen"),
        default="auto",
        help="캡처 방식. auto는 WGC를 먼저 시도하고 실패 시 화면 캡처로 전환",
    )


def open_source(args) -> WindowSource:
    """Open the target given on the command line, or ask the user to pick one."""
    process, title = getattr(args, "process", None), getattr(args, "window", None)
    if not process and not title:
        process, title = pick_target()
    mode = getattr(args, "capture", "auto")
    source = WindowSource(process, title, prefer_wgc=(mode != "screen"))
    if mode == "wgc" and source._mode != "wgc":
        raise SystemExit("WGC로 연결하지 못했습니다. windows-capture 설치 여부를 확인해 주세요")
    name = {"wgc": "WGC(Windows Graphics Capture)", "screen": "화면 캡처"}
    print(f"캡처 방식: {name.get(source._mode, source._mode)}")
    return source
