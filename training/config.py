"""Configuration storage. Slot boxes are kept as ratios of the frame size.

Pixel coordinates would have to be redrawn whenever the window size or the
resolution changes; ratios survive both.
"""

from __future__ import annotations

PROTOCOL = 2  # module interface version; mixing different values is unsafe
FILE_SET = "2026-09-06-a"  # release this file belongs to

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from rules import TEAM_COUNT


@dataclass
class SlotBox:
    """One slot box. All values are ratios between 0 and 1."""

    x0: float
    y0: float
    x1: float
    y1: float

    def pixels(
        self, width: int, height: int, pad_ratio: float = 0.0
    ) -> tuple[int, int, int, int]:
        """Convert the ratio box to pixels.

        Args:
            width: frame width in pixels.
            height: frame height in pixels.

        Returns:
            The box as (x0, y0, x1, y1) in pixels.
        """
        x0 = max(0, int(self.x0 * width))
        x1 = min(width, int(self.x1 * width))
        top, bottom = self.y0 * height, self.y1 * height
        pad = (bottom - top) * pad_ratio
        return x0, max(0, int(top - pad)), x1, min(height, int(bottom + pad))


@dataclass
class Config:
    """Runtime configuration. The target window is not stored here.

    A window title changes whenever the tab or the scene changes, so the target
    is given on the command line or picked from a list at start-up.
    """

    game_area: SlotBox | None = None
    slots: list[SlotBox] = field(default_factory=list)
    team_names: list[str] = field(
        default_factory=lambda: [f"팀 {i + 1}" for i in range(TEAM_COUNT)]
    )
    threshold: int = 190  # a pixel above this brightness counts as ink
    pad_ratio: float = 0.2  # vertical padding as a fraction of the box height
    raise_ratio: float = 0.0  # how far an observed card rises, in box heights

    def area(self, width: int, height: int) -> tuple[int, int, int, int]:
        """Return the pixel rectangle of the game screen inside the frame."""
        if self.game_area is None:
            return 0, 0, width, height
        return self.game_area.pixels(width, height)

    def slot_pixels(
        self, box: SlotBox, width: int, height: int, pad_ratio: float = 0.0
    ) -> tuple[int, int, int, int]:
        """Convert a ratio inside the game screen to frame pixels."""
        ax0, ay0, ax1, ay1 = self.area(width, height)
        x0, y0, x1, y1 = box.pixels(ax1 - ax0, ay1 - ay0, pad_ratio)
        return ax0 + x0, ay0 + y0, ax0 + x1, ay0 + y1

    @staticmethod
    def load(path: str | Path) -> "Config":
        """Load configuration, ignoring keys that are no longer used."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {f.name for f in fields(Config)}
        dropped = sorted(set(data) - known)
        data = {k: v for k, v in data.items() if k in known}
        if dropped:
            print(f"설정에서 사용하지 않는 항목을 건너뜁니다: {', '.join(dropped)}")
        data["slots"] = [SlotBox(**s) for s in data.get("slots", [])]
        area = data.get("game_area")
        data["game_area"] = SlotBox(**area) if area else None
        return Config(**data)

    def save(self, path: str | Path) -> None:
        data = asdict(self)
        Path(path).write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def to_area_ratio(
    box: tuple[float, float, float, float], area: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    """Convert a pixel box to a ratio inside the game screen."""
    ax0, ay0, ax1, ay1 = area
    width, height = max(1e-9, ax1 - ax0), max(1e-9, ay1 - ay0)
    x0, y0, x1, y1 = box
    return (
        (x0 - ax0) / width,
        (y0 - ay0) / height,
        (x1 - ax0) / width,
        (y1 - ay0) / height,
    )


def from_area_ratio(
    ratio: tuple[float, float, float, float], area: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    """Convert a ratio inside the game screen back to a pixel box."""
    ax0, ay0, ax1, ay1 = area
    width, height = ax1 - ax0, ay1 - ay0
    x0, y0, x1, y1 = ratio
    return (ax0 + x0 * width, ay0 + y0 * height, ax0 + x1 * width, ay0 + y1 * height)

