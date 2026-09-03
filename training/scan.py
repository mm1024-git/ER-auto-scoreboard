"""Read the eight team slots from one frame.
"""

from __future__ import annotations

PROTOCOL = 2  # module interface version; mixing different values is unsafe
FILE_SET = "2026-09-06-a"  # release this file belongs to

from dataclasses import dataclass, replace

import numpy as np

from config import Config
from settings import MODEL_MIN_MARGIN, MODEL_MIN_SCORE
from model import SlotReading
REF_SCREEN_WIDTH = 1920.0  # reference frame width used for scaling
from recognize import crop_ink, glyph_spans, ink, line_spans, read_band


@dataclass
class FrameReading:
    """One reading per slot; a covered slot stays None."""

    readings: list[SlotReading | None]
    errors: list[str]
    tops: list[int | None]  # unused for now

    @property
    def complete(self) -> bool:
        return all(r is not None for r in self.readings)

    @property
    def missing(self) -> list[int]:
        return [i for i, r in enumerate(self.readings) if r is None]


def fill_missing(
    got: FrameReading, previous: list[SlotReading] | None
) -> list[SlotReading] | None:
    """Fill covered slots with the last committed values.

    An item popup can cover one or two slots. Rather than dropping the whole
    frame, the covered slots are filled from the last commit and the result is
    validated. If a team was eliminated while covered, the survivor count will
    not match and the frame is rejected anyway.

    Args:
        result: the current scan result.
        last: readings from the last commit, or None.

    Returns:
        A full list of readings, or None when it cannot be completed.
    """
    if got.complete:
        return [r for r in got.readings if r is not None]
    if previous is None:
        return None
    filled: list[SlotReading] = []
    for slot, r in enumerate(got.readings):
        filled.append(r if r is not None else replace(previous[slot], raised=False))
    return filled


def rescale(crop: np.ndarray, factor: float) -> np.ndarray:
    """Resize a patch back to the size used when the samples were made.

    Resizing after thresholding smears the strokes, so the greyscale image is
    scaled first and thresholded afterwards.
    """
    if abs(factor - 1.0) < 0.02 or crop.size == 0:
        return crop
    from PIL import Image

    height, width = crop.shape[:2]
    size = (max(1, round(width * factor)), max(1, round(height * factor)))
    return np.array(Image.fromarray(crop).resize(size, Image.LANCZOS))


def reference_scale(cfg: Config, width: int, height: int) -> float:
    """Return the factor that maps the current frame onto the 1920 px reference."""
    ax0, _, ax1, _ = cfg.area(width, height)
    return REF_SCREEN_WIDTH / max(1, ax1 - ax0)


def slot_crop(
    frame: np.ndarray, cfg: Config, rect: tuple[int, int, int, int]
) -> np.ndarray:
    """Crop one slot and scale it to the reference size.

    Reading and training both use this size. Otherwise the same digit would be
    cropped differently on every resolution and would have to be learned again.
    """
    x0, y0, x1, y1 = rect
    return rescale(frame[y0:y1, x0:x1], reference_scale(cfg, frame.shape[1], frame.shape[0]))


def slot_windows(
    cfg: Config, width: int, height: int, box
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int] | None]:
    """Return the two possible rectangles for a slot: lowered and raised."""
    x0, y0, x1, y1 = cfg.slot_pixels(box, width, height, cfg.pad_ratio)
    lift = int((y1 - y0) * cfg.raise_ratio / (1 + 2 * cfg.pad_ratio))
    if lift <= 0:
        return (x0, y0, x1, y1), None
    return (x0, y0, x1, y1), (x0, max(0, y0 - lift), x1, max(0, y1 - lift))


def _parse_at(
    frame: np.ndarray, rect: tuple[int, int, int, int], cfg: Config, reader
) -> tuple[float, float] | None:
    """Report whether a score can be read at this rectangle."""
    crop = slot_crop(frame, cfg, rect)
    if crop.size == 0:
        return None
    got = read_band(crop, reader, cfg.threshold, MODEL_MIN_SCORE, None, MODEL_MIN_MARGIN)
    return None if got is None else got[0]


def is_score_window(crop: np.ndarray, threshold: int) -> bool:
    """Report whether this rectangle holds a score line."""
    return looks_like_score(crop, threshold)


def looks_like_score(crop: np.ndarray, threshold: int) -> bool:
    """Detect a score line by shape alone, without reading the digits.

    A score line usually carries two decimal points, since TS and KS are both in
    half-point steps. A zero score is printed as "0" with no decimal point, so a
    line split into two groups by one wide gap also counts.
    """
    if crop.size == 0:
        return False
    mask = ink(crop, threshold)
    for y0, y1 in line_spans(mask):
        row = mask[y0:y1]
        line_height = y1 - y0
        pieces = glyph_spans(row)
        if not 2 <= len(pieces) <= 14:
            continue

        dots = 0
        for x0, x1 in pieces:
            shape = crop_ink(row[:, x0:x1]).shape
            if shape[0] <= max(2, line_height // 3) and shape[1] <= max(2, line_height // 2):
                dots += 1
        if dots == 2:
            return True

        if dots == 0 and len(pieces) <= 6:
            # all-zero line: 'TS 0 KS 0' splits on one wide gap
            gaps = [pieces[i + 1][0] - pieces[i][1] for i in range(len(pieces) - 1)]
            if gaps and max(gaps) >= line_height:
                return True
    return False


def detect_observed(
    frame: np.ndarray, cfg: Config, reader=None
) -> int | None:
    """Decide which slot is currently observed.

    A slot that reads a score at the lowered rectangle is not raised. The slot
    that reads nothing there but reads at the raised rectangle is the observed
    one.
    """
    height, width = frame.shape[:2]
    read_hits: list[int] = []
    shape_hits: list[int] = []
    for i, box in enumerate(cfg.slots):
        normal, raised = slot_windows(cfg, width, height, box)
        if raised is None:
            continue
        if reader is not None:
            if _parse_at(frame, normal, cfg, reader) is not None:
                continue
            if _parse_at(frame, raised, cfg, reader) is not None:
                read_hits.append(i)
                continue
        if is_score_window(
            slot_crop(frame, cfg, raised), cfg.threshold
        ) and not is_score_window(slot_crop(frame, cfg, normal), cfg.threshold):
            shape_hits.append(i)

    if read_hits:
        return read_hits[0]
    # Unknown digits stop the score from parsing, but the shape still tells
    # us that this is a score line, so that case is kept.
    return shape_hits[0] if shape_hits else None


def read_slots(
    frame: np.ndarray,
    cfg: Config,
    reader,
    unknown: list[np.ndarray] | None = None,
) -> FrameReading:
    """Read the score line of every slot.

    The observed slot is detected first and read at the raised rectangle; the
    others are read at the lowered one. Only one rectangle is tried per slot, so
    the team number or name next to it is never read by mistake.

    Args:
        frame: captured RGB frame.
        cfg: configuration with slot boxes and thresholds.
        reader: object exposing match/complete/sparse.
        unknown: optional list collecting unreadable glyphs.

    Returns:
        A ScanResult holding the readings and any errors.
    """
    height, width = frame.shape[:2]
    observed = detect_observed(frame, cfg, reader)
    values: list[tuple[float, float] | None] = []
    errors: list[str] = []

    for i, box in enumerate(cfg.slots):
        normal, raised = slot_windows(cfg, width, height, box)
        x0, y0, x1, y1 = raised if (i == observed and raised is not None) else normal
        sink: list = []
        got = read_band(
            slot_crop(frame, cfg, (x0, y0, x1, y1)), reader, cfg.threshold,
            MODEL_MIN_SCORE, sink if unknown is not None else None, MODEL_MIN_MARGIN,
        )
        used = reference_scale(cfg, width, height)
        if unknown is not None and sink:
            for u in sink:
                a, b, c, d = u.rect
                u.rect = (
                    x0 + round(a / used), y0 + round(b / used),
                    x0 + round(c / used), y0 + round(d / used),
                )
            unknown.extend(sink)
        if got is None:
            values.append(None)
            where = "올라간 자리" if i == observed else "내려온 자리"
            errors.append(f"{i + 1}번 팀의 {where}에서 점수 줄을 찾지 못했습니다")
        else:
            values.append(got[0])

    readings: list[SlotReading | None] = [
        None if v is None else SlotReading(ts=v[0], ks=v[1], raised=(i == observed))
        for i, v in enumerate(values)
    ]
    return FrameReading(readings, errors, [])
