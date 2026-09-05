"""Mark the game screen and the score slots. Windows only.

    python regions.py --process EternalReturn.exe

The game screen is dragged once, then one box is drawn per team. Everything is
stored as ratios so the result survives a change of resolution.
"""

from __future__ import annotations

import argparse
import tkinter as tk
from pathlib import Path

import numpy as np

from capture import add_target_args, open_source
from digits import load_reader
from scan import slot_crop
from config import Config, SlotBox, from_area_ratio, to_area_ratio
from settings import CONFIG_PATH, MODEL_MIN_MARGIN, MODEL_MIN_SCORE
from recognize import (
    ink,
    line_spans,
    normalize_boxes,
    read_band,
    read_line,
    to_gray,
)


def _why_not(text: str) -> str:
    """Explain why this text was not accepted as a score."""
    from recognize import SCORE_NUMBER

    if "?" in text:
        unknown_count = text.count("?")
        return f"미판별 글자 {unknown_count}개 포함"
    found = SCORE_NUMBER.findall(text)
    if len(found) < 2:
        return f"0.5 단위 수가 {len(found)}개입니다 (두 개여야 합니다)"
    return f"추출된 수: {found[-2]}, {found[-1]}"


def _more_useful(text: str, best: str) -> bool:
    """Choose which of two lines to display.

    Preferring the longer one would favour the team name, since nine question
    marks beat a score line. The line with more digits wins, then the one with
    fewer question marks.
    """
    def score(s: str) -> tuple[int, int, int]:
        return (sum(ch.isdigit() for ch in s), -s.count("?"), len(s))

    return score(text) > score(best)


class RegionEditor:
    HELP = (
        "게임 화면 지정을 누른 뒤 게임 화면 전체를 한 번 드래그하고, 팀마다 점수 글자에 "
        "맞춰 상자를 지정합니다. 옵저빙으로 올라간 칸은 올라간 위치에 그대로 지정합니다. "
        "실선이 내려온 위치, 점선이 올라간 위치입니다."
    )

    def __init__(self, frame: np.ndarray, cfg: Config, config_path: str, reader):
        from PIL import Image, ImageTk

        self.frame = frame
        self.cfg = cfg
        self.config_path = config_path
        self.reader = reader
        self.mode = "slots"

        h, w = frame.shape[:2]
        self.scale = min(1.0, 1600 / w, 900 / h)
        view = Image.fromarray(frame).resize((int(w * self.scale), int(h * self.scale)))

        self.root = tk.Tk()
        self.root.title("게임 화면과 점수 칸 지정")

        bar = tk.Frame(self.root)
        bar.pack(fill="x")
        self.area_button = tk.Button(bar, text="게임 화면 지정", command=self.toggle_area_mode)
        self.area_button.pack(side="left")
        tk.Button(bar, text="선택 삭제", command=self.delete_selected).pack(side="left")
        tk.Button(bar, text="전부 지우기", command=self.clear).pack(side="left")
        tk.Button(bar, text="저장", command=self.save).pack(side="left")
        tk.Button(bar, text="닫기", command=self.root.destroy).pack(side="right")

        self.status = tk.Label(self.root, text=self.HELP, anchor="w", justify="left")
        self.status.pack(fill="x")

        self.photo = ImageTk.PhotoImage(view)
        self.canvas = tk.Canvas(self.root, width=view.width, height=view.height)
        self.canvas.pack()
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)

        self.area: tuple[float, float, float, float] | None = (
            cfg.game_area.pixels(w, h) if cfg.game_area else None
        )
        self.boxes: list[tuple[float, float, float, float]] = [
            cfg.slot_pixels(b, w, h) for b in cfg.slots
        ]
        self.selected: int | None = None
        self.start: tuple[int, int] | None = None
        self.rubber = None

        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        # key names differ under a Korean IME, so key codes are checked as well
        self.root.bind("<Key>", self.on_key)
        self.redraw()

    # ---- target area -----------------------------------------------------

    def area_rect(self) -> tuple[float, float, float, float]:
        if self.area is not None:
            return self.area
        h, w = self.frame.shape[:2]
        return (0.0, 0.0, float(w), float(h))

    def set_area(self, box: tuple[float, float, float, float]) -> None:
        """Move the slot boxes when the game screen is redrawn.

        Slots are ratios inside the game screen, so they follow when it grows
        or shrinks.
        """
        old = self.area_rect()
        ratios = [to_area_ratio(b, old) for b in self.boxes]
        self.area = box
        self.boxes = [from_area_ratio(r, box) for r in ratios]

    def toggle_area_mode(self) -> None:
        self.mode = "area" if self.mode == "slots" else "slots"
        self.area_button.config(
            text="점수 칸 지정으로" if self.mode == "area" else "게임 화면 지정"
        )
        self.redraw()

    # ---- interaction -----------------------------------------------------

    def on_key(self, event) -> None:
        key = (event.keysym or "").lower()
        if key == "delete" or event.keycode == 46:
            self.delete_selected()
        elif key == "s" or event.keycode == 83:
            self.save()

    def on_press(self, event) -> None:
        if self.mode == "slots":
            hit = self.hit_test(event.x, event.y)
            if hit is not None:
                self.selected = hit
                self.redraw()
                return
            self.selected = None
        self.start = (event.x, event.y)

    def on_drag(self, event) -> None:
        if self.start is None:
            return
        if self.rubber is not None:
            self.canvas.delete(self.rubber)
        color = "#5aa9ff" if self.mode == "area" else "#39d353"
        self.rubber = self.canvas.create_rectangle(
            self.start[0], self.start[1], event.x, event.y, outline=color
        )

    def on_release(self, event) -> None:
        if self.start is None:
            return
        x0, y0 = self.start
        x1, y1 = event.x, event.y
        self.start = None
        if abs(x1 - x0) < 5 or abs(y1 - y0) < 5:
            self.redraw()
            return
        box = tuple(
            v / self.scale
            for v in (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
        )
        if self.mode == "area":
            self.set_area(box)
            self.toggle_area_mode()
            return
        self.boxes.append(box)
        self.boxes.sort()
        self.redraw()

    def hit_test(self, sx: int, sy: int) -> int | None:
        for i, (x0, y0, x1, y1) in enumerate(self.boxes):
            if x0 * self.scale <= sx <= x1 * self.scale and y0 * self.scale <= sy <= y1 * self.scale:
                return i
        return None

    def delete_selected(self) -> None:
        if self.selected is not None:
            del self.boxes[self.selected]
            self.selected = None
            self.redraw()

    def clear(self) -> None:
        self.boxes = []
        self.selected = None
        self.redraw()

    # ---- preview ---------------------------------------------------------

    def read_box(self, box: tuple[float, float, float, float], slot: int = 0) -> str:
        """Read the box exactly as the aggregator would, at both positions.

        The text that was read and the reason it was rejected are printed to the
        console, since an X in the box does not say what went wrong.
        """
        if self.reader is None:
            return ""
        h, w = self.frame.shape[:2]
        x0, y0, x1, y1 = (int(v) for v in box)
        pad = int((y1 - y0) * self.cfg.pad_ratio)
        lift = int((y1 - y0) * self.cfg.raise_ratio)
        partial = ""
        for up in (0, lift):
            rect = (
                max(0, x0), max(0, y0 - pad - up),
                min(w, x1), min(h, y1 + pad - up),
            )
            # The same path as the aggregator: the crop is scaled to the reference
            # size and the brightness is passed along. Without it only the mask
            # would be compared and everything would read as "?".
            crop = slot_crop(self.frame, self.cfg, rect)
            if crop.size == 0:
                continue
            trace: list[str] = []
            got = read_band(
                crop, self.reader, self.cfg.threshold, MODEL_MIN_SCORE, None,
                MODEL_MIN_MARGIN, trace,
            )
            if got is not None:
                (ts, ks), _ = got
                return f"{ts}/{ks}" + ("↑" if up else "")
            place_name = "올라간 자리" if up else "내려온 자리"
            for line in trace:  # what the aggregator path actually saw
                print(f"  {slot + 1}번 칸 {place_name} 집계 경로 {line}")
            # Show how far the reading got: a "?" marks a glyph that needs a
            # label, an empty result means the box holds no glyph.
            mask, gray = ink(crop, self.cfg.threshold), to_gray(crop)
            for a, b in line_spans(mask):
                glyph_trace: list[str] = []
                text = read_line(
                    mask[a:b], self.reader, MODEL_MIN_SCORE, None, (0, a),
                    MODEL_MIN_MARGIN, gray[a:b], glyph_trace,
                )
                for line in glyph_trace:
                    print(f"  {slot + 1}번 칸 {place_name} 미리보기 {line}")
                print(
                    f"  {slot + 1}번 칸 {place_name}: 읽은 글자 {text!r} "
                    f"-> {_why_not(text)}"
                )
                if _more_useful(text, partial):
                    partial = text
        return f"X {partial}".strip()

    # ---- drawing ---------------------------------------------------------

    def redraw(self) -> None:
        self.canvas.delete("box")
        if self.rubber is not None:
            self.canvas.delete(self.rubber)
            self.rubber = None

        if self.area is not None:
            x0, y0, x1, y1 = self.area
            self.canvas.create_rectangle(
                x0 * self.scale, y0 * self.scale, x1 * self.scale, y1 * self.scale,
                outline="#5aa9ff", width=2, tags="box",
            )

        readable = 0
        for i, (x0, y0, x1, y1) in enumerate(self.boxes):
            lift = (y1 - y0) * self.cfg.raise_ratio
            color = "#ff5555" if i == self.selected else "#39d353"
            self.canvas.create_rectangle(
                x0 * self.scale, y0 * self.scale, x1 * self.scale, y1 * self.scale,
                outline=color, width=2, tags="box",
            )
            if lift > 0:
                self.canvas.create_rectangle(
                    x0 * self.scale, (y0 - lift) * self.scale,
                    x1 * self.scale, (y1 - lift) * self.scale,
                    outline=color, dash=(3, 3), tags="box",
                )
            got = self.read_box((x0, y0, x1, y1), i)
            if got and got != "X":
                readable += 1
            self.canvas.create_text(
                x0 * self.scale + 4, y0 * self.scale - 8,
                text=f"{i + 1} {got}".strip(), fill=color, anchor="w", tags="box",
            )

        area_note = "게임 화면이 지정되지 않아 프레임 전체를 사용합니다." if self.area is None else ""
        lift_note = (
            "상승 폭이 아직 측정되지 않았습니다. 옵저빙으로 올라간 칸을 해당 위치에 지정한 뒤 저장하면 측정됩니다."
            if self.cfg.raise_ratio <= 0
            else f"상승 폭은 상자 높이의 {self.cfg.raise_ratio:.2f}배입니다."
        )
        if self.reader is None:
            read_note = "학습된 모델이 없어 판독 미리보기를 사용할 수 없습니다."
        else:
            read_note = f"상자 {len(self.boxes)}개 중 {readable}개가 판별됩니다."
        mode_note = "현재 게임 화면 지정 중입니다." if self.mode == "area" else ""
        self.status.config(
            text="\n".join(x for x in (self.HELP, mode_note, area_note, read_note, lift_note) if x)
        )

    # ---- saving ----------------------------------------------------------

    def save(self) -> None:
        """Align every box to the lowered position and save them as screen ratios."""
        h, w = self.frame.shape[:2]
        if self.boxes:
            layout = normalize_boxes([tuple(int(v) for v in b) for b in self.boxes])
            self.boxes = [tuple(float(v) for v in b) for b in layout.boxes]
            if layout.raise_ratio > 0:
                self.cfg.raise_ratio = layout.raise_ratio

        ax0, ay0, ax1, ay1 = self.area_rect()
        aw, ah = max(1.0, ax1 - ax0), max(1.0, ay1 - ay0)
        self.cfg.game_area = (
            None if self.area is None else SlotBox(x0=ax0 / w, y0=ay0 / h, x1=ax1 / w, y1=ay1 / h)
        )
        self.cfg.slots = [
            SlotBox(
                x0=(x0 - ax0) / aw, y0=(y0 - ay0) / ah,
                x1=(x1 - ax0) / aw, y1=(y1 - ay0) / ah,
            )
            for x0, y0, x1, y1 in self.boxes
        ]
        self.cfg.save(self.config_path)
        self.redraw()
        self.status.config(
            text=f"{len(self.cfg.slots)}칸을 {self.config_path}에 저장했습니다. "
            f"상승 폭은 상자 높이의 {self.cfg.raise_ratio:.2f}배입니다."
        )

    def run(self) -> None:
        self.root.mainloop()


def _check_file_set() -> None:
    """Stop when the local modules come from different releases."""
    from recognize import check_file_set

    check_file_set()


def main() -> None:
    _check_file_set()
    parser = argparse.ArgumentParser(description="게임 화면과 점수 칸 지정")
    parser.add_argument("--config", default=CONFIG_PATH)
    add_target_args(parser)
    args = parser.parse_args()

    path = Path(args.config)
    cfg = Config.load(path) if path.exists() else Config()

    source = open_source(args)
    try:
        frame = source.first_frame()
    finally:
        source.close()

    reader = None
    try:
        reader, _, _ = load_reader(cfg)
    except SystemExit:
        pass  # areas can be marked even without a reader
    RegionEditor(frame, cfg, str(path), reader).run()


if __name__ == "__main__":
    main()
