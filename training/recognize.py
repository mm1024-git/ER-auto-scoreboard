"""Read the digits on the scoreboard.

A general OCR engine is not used. The score glyphs have a fixed font, size and
bright colour, so thresholding by brightness and comparing against stored glyph
images is both faster and less error prone.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# release marker; the entry points compare it to catch mixed releases
PROTOCOL = 2  # module interface version; mixing different values is unsafe
FILE_SET = "2026-09-04-i"

CHARS = "0123456789."  # labels are not read, so only digits and the dot are kept

# Glyph canvas, measured on a 1920 px frame. A digit is 8 to 10 px wide and the
# slot calculation adds the gap, giving 9 to 11; the widest value is used. A
# narrower canvas smears the strokes and makes 1 look like 6. Height is 18 px.
TEMPLATE_WIDTH = 11
TEMPLATE_HEIGHT = 18
EDGE_MARGIN = 1  # extra pixels taken on both sides of a glyph
STROKE_LEVEL = 0.5  # a pixel above the midpoint between background and peak is ink
from settings import INCOMPLETE_SCORE
_FILE_NAME = {".": "dot"}

# Scores come in half points, so the first decimal is 0 or 5. Two such numbers
# in a row are TS then KS. The TS and KS labels are darker than the digits and
# disappear with the threshold, so they are not read.
# A zero score is printed as '0', not '0.0', so numbers without a decimal
# point are accepted as well.
SCORE_NUMBER = re.compile(r"[0-9]+(?:\.[05])?")


def check_shadowed_modules(folder: str | None = None) -> None:
    """Check whether a local .py shadows a standard library module.

    A file named select.py, for example, is imported instead of the standard
    select module and breaks http.server with an unrelated message.

    Args:
        folder: directory to inspect; defaults to this file's directory.

    Raises:
        SystemExit: at least one file shadows a standard module.
    """
    import sys
    from pathlib import Path

    place = Path(folder) if folder else Path(__file__).resolve().parent
    stdlib_names = set(sys.stdlib_module_names) - {"this", "antigravity"}
    shadowed = sorted(
        path.stem
        for path in place.glob("*.py")
        if path.stem in stdlib_names
    )
    if shadowed:
        raise SystemExit(
            "파이썬 표준 모듈과 이름이 같은 파일이 있습니다: "
            + ", ".join(f"{name}.py" for name in shadowed)
            + ". 해당 파일이 표준 모듈을 가려 다른 위치에서 오류가 발생합니다. "
            "파일 이름을 변경하거나 삭제해 주세요."
        )


def check_file_set(expected: str | int = PROTOCOL) -> None:
    """Check that the local modules belong to one compatible release.

    FILE_SET records when a file last changed and may differ between files. Only
    PROTOCOL, which changes when the module interfaces change, has to match.

    Raises:
        SystemExit: a module carries a different protocol version.
    """
    import importlib

    check_shadowed_modules()

    mismatched: list[str] = []
    for name in (
        "recognize", "capture", "scan", "model", "config", "digits", "rules",
        "overlay", "server", "history", "settings",
    ):
        try:
            mod = importlib.import_module(name)
        except ImportError:
            continue
        version = getattr(mod, "PROTOCOL", None)
        if version != PROTOCOL:
            shown = version if version is not None else "표시 없음"
            mismatched.append(f"{name}.py는 {shown}판")
    if mismatched:
        raise SystemExit(
            "폴더 안의 .py 파일 판이 서로 다릅니다. "
            f"현재 판은 {PROTOCOL}이며 " + ", ".join(mismatched) + "입니다. "
            "전달받은 파일을 모두 덮어썼는지 확인해 주세요."
        )


def ink(img: np.ndarray, threshold: int = 190) -> np.ndarray:
    """Return a mask of the bright pixels. img is (H, W, 3) or (H, W, 4)."""
    rgb = img[:, :, :3].astype(np.int16)
    return rgb.min(axis=2) >= threshold


def _spans(flags: np.ndarray, gap: int = 0) -> list[tuple[int, int]]:
    """Return the runs of True as [start, end) spans, bridging gaps up to gap."""
    out: list[tuple[int, int]] = []
    start = None
    hole = 0
    for i, v in enumerate(flags):
        if v:
            if start is None:
                start = i
            hole = 0
        elif start is not None:
            hole += 1
            if hole > gap:
                out.append((start, i - hole + 1))
                start = None
    if start is not None:
        out.append((start, len(flags) - hole))
    return out


def line_spans(mask: np.ndarray, gap: int = 1, min_height: int = 4) -> list[tuple[int, int]]:
    rows = mask.any(axis=1)
    return [(a, b) for a, b in _spans(rows, gap) if b - a >= min_height]


def glyph_spans(mask: np.ndarray, gap: int = 0, min_width: int = 1) -> list[tuple[int, int]]:
    """Split glyphs on a single empty column.

    Score digits are about eight pixels tall with a one pixel gap, so bridging
    one column would read '2.5' as a single blob.
    """
    cols = mask.any(axis=0)
    return [(a, b) for a, b in _spans(cols, gap) if b - a >= min_width]


def crop_ink(mask: np.ndarray) -> np.ndarray:
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        return np.zeros((1, 1), dtype=bool)
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def to_gray(img: np.ndarray) -> np.ndarray:
    """Convert an image to a 0..1 brightness array using the darkest channel.

    An array that is already in 0..1 is returned unchanged; dividing it again
    would wash the glyph out.
    """
    if img.dtype == bool:
        return img.astype(np.float32)
    plane = img[:, :, :3].min(axis=2) if img.ndim == 3 else img
    value = plane.astype(np.float32)
    return value / 255.0 if float(value.max()) > 1.0 else value


def _own_glyph(mask: np.ndarray) -> np.ndarray:
    """Keep only the glyph that belongs to this slot.

    Slot rectangles are computed from the right edge, so a neighbouring glyph can
    reach one or two columns into the crop and distort the comparison.
    """
    height, width = mask.shape
    seen = np.zeros((height, width), dtype=bool)
    blobs: list[tuple[int, list[tuple[int, int]]]] = []
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or seen[y, x]:
                continue
            stack = [(y, x)]
            seen[y, x] = True
            pixels = []
            while stack:
                cy, cx = stack.pop()
                pixels.append((cy, cx))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < height and 0 <= nx < width:
                            if mask[ny, nx] and not seen[ny, nx]:
                                seen[ny, nx] = True
                                stack.append((ny, nx))
            blobs.append((len(pixels), pixels))

    if len(blobs) <= 1:
        return mask
    largest = max(n for n, _ in blobs)
    kept = np.zeros_like(mask)
    for n, pixels in blobs:
        if n >= largest * 0.25:
            for y, x in pixels:
                kept[y, x] = True
    return kept


def canonical(
    patch: np.ndarray, mask: np.ndarray | None = None, already: bool = False
) -> np.ndarray:
    """Fit one glyph into the fixed canvas.

    A crop varies by a few pixels with the resolution and the cropping method.
    Comparing without normalising would score the same digit lower.
    """
    from PIL import Image

    if already:
        return patch.astype(np.float32)  # already normalised to the canvas

    gray = to_gray(patch)
    if gray.size == 0:
        return np.zeros((TEMPLATE_HEIGHT, TEMPLATE_WIDTH), dtype=np.float32)

    # Only the bright pixels are kept: comparing raw brightness would let the
    # density of faint strokes swing the score for the same digit.
    if mask is not None:
        gray = np.where(mask, 1.0, 0.0).astype(np.float32)
    else:
        gray = (gray >= 0.7 * float(gray.max())).astype(np.float32)

    if mask is not None:
        mask = _own_glyph(mask)

    # The background is dropped so that half the compared image is not empty,
    # which would make different digits look alike. The ink mask defines the
    # bounds, so any crop of the same digit yields the same image; one pixel is
    # added to keep faint stroke edges.
    bright = mask if mask is not None else gray >= 0.7 * float(gray.max())
    rows = np.flatnonzero(bright.any(axis=1))
    cols = np.flatnonzero(bright.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        return np.zeros((TEMPLATE_HEIGHT, TEMPLATE_WIDTH), dtype=np.float32)
    top = max(0, int(rows[0]) - EDGE_MARGIN)
    bottom = min(gray.shape[0], int(rows[-1]) + 1 + EDGE_MARGIN)
    left = max(0, int(cols[0]) - EDGE_MARGIN)
    right = min(gray.shape[1], int(cols[-1]) + 1 + EDGE_MARGIN)
    gray = gray[top:bottom, left:right]

    # Height is scaled to the canvas and the aspect ratio is kept: stretching the
    # width would widen a narrow 1 into a 6. Brightness is not thresholded, since
    # keeping only white pixels erases faint strokes.
    height, width = gray.shape
    target = max(1, min(TEMPLATE_WIDTH, round(width * TEMPLATE_HEIGHT / height)))
    img = Image.fromarray(np.clip(gray * 255, 0, 255).astype(np.uint8))
    resized = np.asarray(
        img.resize((target, TEMPLATE_HEIGHT), Image.LANCZOS), dtype=np.float32
    ) / 255.0

    out = np.zeros((TEMPLATE_HEIGHT, TEMPLATE_WIDTH), dtype=np.float32)
    left = (TEMPLATE_WIDTH - target) // 2
    out[:, left : left + target] = resized
    top = float(out.max())
    return out / top if top > 0 else out


def _pad_to(a: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    out[: a.shape[0], : a.shape[1]] = a[: shape[0], : shape[1]]
    return out


def _agreement(x: np.ndarray, y: np.ndarray) -> float:
    """Return intersection over union: 1 for identical, 0 for no overlap."""
    a, b = x > 0.5, y > 0.5
    union = np.count_nonzero(a | b)
    if union == 0:
        return 0.0
    return np.count_nonzero(a & b) / union


def similarity(a: np.ndarray, b: np.ndarray, slack: int = 1) -> float:
    """Return how similar two glyphs are, tolerating a shift of one or two pixels.

    Raw brightness is compared; keeping only white pixels would erase faint
    strokes and drop the score for the same digit.
    """
    x = a.astype(np.float32)
    y = b.astype(np.float32)
    shape = (
        max(x.shape[0], y.shape[0]) + 2 * slack,
        max(x.shape[1], y.shape[1]) + 2 * slack,
    )
    base = np.zeros(shape, dtype=np.float32)
    base[slack : slack + x.shape[0], slack : slack + x.shape[1]] = x

    best = 0.0
    for dy in range(-slack, slack + 1):
        for dx in range(-slack, slack + 1):
            moved = np.zeros(shape, dtype=np.float32)
            top, left = slack + dy, slack + dx
            moved[top : top + y.shape[0], left : left + y.shape[1]] = y
            best = max(best, _agreement(base, moved))
    return best


@dataclass
class Unknown:
    """An unread glyph patch together with the rectangle it came from."""

    patch: np.ndarray
    rect: tuple[int, int, int, int]


def read_line(
    mask: np.ndarray,
    reader,
    min_score: float = 0.70,
    unknown: list[Unknown] | None = None,
    origin: tuple[int, int] = (0, 0),
    min_margin: float = 0.10,
    gray: np.ndarray | None = None,
    trace: list[str] | None = None,
) -> str:
    """Read a line from the left and return it as text; unread glyphs become '?'.

    When a list is passed as unknown, the unread patches are collected in it.
    """
    out = []
        # Glyph height is the median height of the individual patches. Measuring the
    # whole ink extent would return a value much larger than a glyph when two
    # lines fall into the box, hiding the wide gap at the label.
    heights = [crop_ink(mask[:, a:b]).shape[0] for a, b in glyph_spans(mask)]
    if heights:
        line_height = int(np.median([h for h in heights if h > 0] or [1]))
    else:
        used_rows = np.flatnonzero(mask.any(axis=1))
        line_height = int(used_rows[-1] - used_rows[0] + 1) if used_rows.size else mask.shape[0]
    threshold = max(min_score, INCOMPLETE_SCORE) if reader.sparse else min_score
    prev_end = None
    for x0, x1 in glyph_spans(mask):
        # A wide gap, such as the label area, becomes a space: an all-zero line
        # would otherwise read 'TS 0 KS 0' as '00', that is one number.
        if prev_end is not None and x0 - prev_end >= line_height:
            out.append(" ")
        prev_end = x1
        patch = mask[:, x0:x1]
        a = max(0, x0 - EDGE_MARGIN)
        b = min(mask.shape[1], x1 + EDGE_MARGIN)
        bright = patch if gray is None else gray[:, a:b]
        wide = mask[:, a:b]
        shape = crop_ink(patch).shape
        # A decimal point is only a couple of pixels, so overlap scoring rates it
        # low; a patch below a third of the line height and half its width is a dot.
        if shape[0] <= max(2, line_height // 3) and shape[1] <= max(2, line_height // 2):
            out.append(".")
            continue
        ch, score, other = reader.match(bright, patch if gray is None else wide)
        if trace is not None:
            trace.append(
                f"[{x0}-{x1}] {ch} {score:.2f} (2등 {other:.2f}, threshold {threshold:.2f}, "
                f"차이기준 {min_margin:.2f}, piece {bright.shape[0]}x{bright.shape[1]})"
            )
        if score >= threshold and score - other >= min_margin:
            out.append(ch)
        else:
            out.append("?")
            if unknown is not None:
                ox, oy = origin
                unknown.append(
                    Unknown(crop_ink(patch), (ox + x0, oy, ox + x1, oy + mask.shape[0]))
                )
    return "".join(out)


def parse_score(text: str) -> tuple[float, float] | None:
    """Extract the two scores from the read text, or None when there are not two.

    In '2.52.5' the first 2.5 is TS and the second is KS. A zero line has no
    decimal point and arrives as '0 0', split on the label gap.
    """
    found = SCORE_NUMBER.findall(text)
    if len(found) < 2:
        return None
    if "?" in text:
        return None  # an unread glyph makes the neighbouring numbers unreliable
    # Debris at the label can read as a digit. The score line is aligned to the
    # right edge, so the last two numbers are TS and KS.
    return float(found[-2]), float(found[-1])


def read_band(
    img: np.ndarray,
    reader,
    threshold: int = 190,
    min_score: float = 0.70,
    unknown: list[Unknown] | None = None,
    min_margin: float = 0.10,
    trace: list[str] | None = None,
) -> tuple[tuple[float, float], int] | None:
    """Find the score line inside a generously sized band.

    An observed team's card moves up, so the vertical position is not fixed; the
    band is searched for a line that reads as 'TS...KS...'.
    """
    mask = ink(img, threshold)
    lines = line_spans(mask)

    # Each line is passed on its own. Passing the whole box brings one extra row
    # above and below the glyphs, which differs from the training crops and stops
    # the same digit from being recognised.
    gray = to_gray(img)
    for y0, y1 in lines:
        glyph_trace: list[str] | None = [] if trace is not None else None
        text = read_line(
            mask[y0:y1], reader, min_score, unknown, (0, y0), min_margin,
            gray[y0:y1], glyph_trace,
        )
        if trace is not None:
            trace.append(f"{y0}~{y1}line: {text!r}")
            trace.extend(f"    {line}" for line in (glyph_trace or []))
        parsed = parse_score(text)
        if parsed is not None:
            return parsed, y0

    text = read_line(mask, reader, min_score, unknown, (0, 0), min_margin, gray)
    if trace is not None:
        trace.append(f"entry 전체: {text!r}")
    parsed = parse_score(text)
    if parsed is not None:
        return parsed, lines[0][0] if lines else 0
    return None


def _line_groups(
    mask: np.ndarray, y0: int, y1: int, gap: int
) -> list[tuple[int, int, int, int]]:
    """Split a line into blobs wherever the horizontal gap is at least gap.

    Very narrow blobs are discarded as debris, using half the line height rather
    than a pixel count so that the rule scales with the resolution.
    """
    row = mask[y0:y1]
    min_width = max(2, (y1 - y0) // 2)
    return [
        (x0, y0, x1, y1)
        for x0, x1 in glyph_spans(row, gap=gap)
        if x1 - x0 >= min_width
    ]


def _pitch_error(boxes: list[tuple[int, int, int, int]]) -> float:
    """Return how evenly the blobs are spaced; smaller is more even."""
    centers = sorted((b[0] + b[2]) / 2 for b in boxes)
    gaps = [b - a for a, b in zip(centers, centers[1:])]
    if not gaps:
        return float("inf")
    mean = sum(gaps) / len(gaps)
    if mean <= 0:
        return float("inf")
    return max(abs(g - mean) for g in gaps) / mean


def _best_grouping(
    mask: np.ndarray,
    y0: int,
    y1: int,
    reader,
    count: int,
    max_pitch_error: float,
    min_score: float = 0.70,
    gray: np.ndarray | None = None,
) -> tuple[float, list[tuple[int, int, int, int]]] | None:
    """Split a line at several gap widths and keep the most even set of eight.

    The difference between the gap inside a glyph and the gap between slots
    varies with the resolution, so one fixed width cannot be used.
    """
    best: tuple[float, list[tuple[int, int, int, int]]] | None = None
    for gap in range(2, 61, 2):
        groups = _line_groups(mask, y0, y1, gap)
        if reader is not None:
            groups = [
                g
                for g in groups
                if parse_score(
                    read_line(
                        mask[g[1] : g[3], g[0] : g[2]], reader, min_score,
                        gray=None if gray is None else gray[g[1] : g[3], g[0] : g[2]],
                    )
                )
                is not None
            ]
        if len(groups) != count:
            continue
        error = _pitch_error(groups)
        if error <= max_pitch_error and (best is None or error < best[0]):
            best = (error, groups)
    return best


@dataclass
class Layout:
    """Detected scoreboard layout.

    boxes are the rectangles for the lowered cards; raise_ratio is how far an
    observed card rises, in box heights.
    """

    boxes: list[tuple[int, int, int, int]]
    raise_ratio: float


def normalize_boxes(
    boxes: list[tuple[int, int, int, int]], lift_floor: float = 0.3
) -> Layout:
    """Derive the two rectangles per slot from eight boxes.

    Boxes at the common height are the lowered positions; a box drawn clearly
    above them marks the raised position.
    """
    if not boxes:
        return Layout([], 0.0)

    tops = sorted(b[1] for b in boxes)
    heights = sorted(b[3] - b[1] for b in boxes)
    normal_top = tops[len(tops) // 2]
    box_height = max(1, heights[len(heights) // 2])

    lifts = [normal_top - t for t in tops if normal_top - t > box_height * lift_floor]
    lift = sorted(lifts)[len(lifts) // 2] if lifts else 0
    aligned = [
        (b[0], normal_top, b[2], normal_top + box_height) for b in sorted(boxes)
    ]
    return Layout(aligned, lift / box_height)


def _tight_box(
    mask: np.ndarray, group: tuple[int, int, int, int]
) -> tuple[int, int, int, int]:
    """Shrink a rectangle to the rows the blob actually occupies."""
    x0, y0, x1, y1 = group
    sub = mask[y0:y1, x0:x1]
    rows = np.flatnonzero(sub.any(axis=1))
    if rows.size == 0:
        return group
    return x0, y0 + int(rows[0]), x1, y0 + int(rows[-1]) + 1


def autodetect_slots(
    img: np.ndarray,
    reader=None,
    band_top: float = 0.55,
    threshold: int = 190,
    count: int = 8,
    max_pitch_error: float = 0.25,
) -> Layout:
    """Find the eight score slots and the raise ratio in the lower part of a frame."""
    height, _ = img.shape[:2]
    top = int(height * band_top)
    mask = ink(img[top:], threshold)

    lines = line_spans(mask)
    ranges: list[tuple[int, int]] = list(lines)
    # An observed slot sits on its own raised line, so the union of two nearby
    # lines is offered as a candidate too.
    for (a0, a1), (b0, b1) in zip(lines, lines[1:]):
        if b0 - a1 <= max(a1 - a0, b1 - b0):
            ranges.append((a0, b1))

    candidates: list[tuple[int, list[tuple[int, int, int, int]]]] = []
    band_gray = to_gray(img[top:])
    for y0, y1 in ranges:
        found = _best_grouping(
            mask, y0, y1, reader, count, max_pitch_error, gray=band_gray
        )
        if found is not None:
            candidates.append((y0, found[1]))

    if not candidates:
        return Layout([], 0.0)

    _, best = min(candidates, key=lambda c: c[0])  # topmost line
    tight = [_tight_box(mask, g) for g in sorted(best)]
    layout = normalize_boxes(tight)
    return Layout(
        [(b[0], top + b[1], b[2], top + b[3]) for b in layout.boxes],
        layout.raise_ratio,
    )
