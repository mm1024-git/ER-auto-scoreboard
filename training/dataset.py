"""Build the training dataset from a folder of screenshots.

    python dataset.py --config config.json --shots shots --out dataset

Every screenshot is cropped slot by slot, at both the lowered and the raised
position, and each glyph is stored as one patch. Labels come from a hand written
truth file if present, then from OCR, and anything left is put aside for manual
labelling.
"""

from __future__ import annotations

FILE_SET = "2026-09-04-g"  # release this file belongs to

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from config import Config
from settings import (
    MODEL_PATH,
    CONFIG_PATH,
    DATASET_DIR,
    DIGIT_LIKE,
    OTHER_CAP,
    SHOTS_DIR,
    TESSERACT_ENV,
    TESSERACT_PLACES,
)
from labeling import glyphs_in
from model import SlotReading, interpret
from rules import ScanError
from recognize import (
    EDGE_MARGIN,
    SCORE_NUMBER,
    canonical,
    glyph_spans,
    ink,
    read_line,
    to_gray,
)
from scan import is_score_window, reference_scale, slot_crop, slot_windows

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp")


@dataclass
class Sample:
    """One cut patch together with its label."""

    form: np.ndarray  # image normalised to the canvas
    label: str  # "0"~"9", "dot", "other", "unlabeled"
    shot: str
    slot: int
    where: str  # "down" or "up"
    x0: int
    x1: int
    context: np.ndarray | None = None  # context image shown to the operator


LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"




def find_tesseract(given: str | None = None) -> str | None:
    """Locate the Tesseract executable, or return None.

    The search order is the path given on the command line, the TESSERACT_CMD
    environment variable, PATH, and the usual Windows install locations, so the
    user never has to edit the code.
    """
    import os
    from shutil import which

    candidates = [given, os.environ.get(TESSERACT_ENV)]
    candidates += [which("tesseract"), which("tesseract.exe")]
    candidates += [os.path.expandvars(place) for place in TESSERACT_PLACES]
    for place in candidates:
        if place and Path(place).exists():
            return str(place)
    return None


def ocr_ready(given: str | None = None) -> tuple[bool, str]:
    """Report whether OCR is usable, with the reason when it is not.

    Without this check OCR could silently fail and every patch would end up
    unlabelled with no explanation.
    """
    try:
        import pytesseract
    except ImportError:
        return False, "pytesseract가 설치되어 있지 않습니다. pip install pytesseract 로 설치해 주세요"

    place = find_tesseract(given)
    if place:
        pytesseract.pytesseract.tesseract_cmd = place
    try:
        version = pytesseract.get_tesseract_version()
    except Exception as e:
        searched = "\n    ".join(
            [str(given or "(--tesseract 미지정)"), f"환경변수 {TESSERACT_ENV}", "PATH"]
            + [place for place in TESSERACT_PLACES]
        )
        return False, (
            f"Tesseract 본체를 찾지 못했다({e}).\n"
            "  우분투는 apt install tesseract-ocr, 윈도우는 UB-Mannheim 배포본을 설치한다.\n"
            "  설치했음에도 찾지 못하는 경우 실행 파일 경로를 다음과 같이 지정한다.\n"
            '    python dataset.py --tesseract "C:\\Program Files\\Tesseract-OCR\\tesseract.exe" ...\n'
            f"  찾아본 곳:\n    {searched}"
        )
    return True, f"Tesseract {version} ({place or 'PATH에서 찾음'})"


def ocr_chars(
    crop: np.ndarray, threshold: int, whitelist: str, target_height: int = 40
) -> list[tuple[str, float, float]]:
    """Read a whole line and return the characters with their x positions.

    Tesseract barely reads single glyphs, at any scale, so the line is read as a
    whole and the results are mapped back onto the patches.
    """
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        return []

    mask = ink(crop, threshold)
    used_rows = np.flatnonzero(mask.any(axis=1))
    if used_rows.size == 0:
        return []
    line_height = int(used_rows[-1] - used_rows[0] + 1)
    scale = max(1, round(target_height / max(1, line_height)))
    margin = 20

    big = np.kron((mask * 255).astype(np.uint8), np.ones((scale, scale), dtype=np.uint8))
    big = np.pad(big, margin)
    image = Image.fromarray(255 - big)  # black text on white background
    try:
        boxes = pytesseract.image_to_boxes(
            image, config=f"--psm 7 -c tessedit_char_whitelist={whitelist}"
        )
    except Exception:
        return []

    out: list[tuple[str, float, float]] = []
    for line in boxes.strip().splitlines():
        piece = line.split()
        if len(piece) < 5:
            continue
        ch, x0, x1 = piece[0], float(piece[1]), float(piece[3])
        out.append((ch, (x0 - margin) / scale, (x1 - margin) / scale))
    return out


def labels_by_ocr(
    crop: np.ndarray, pieces: list, cfg: Config, score_line: bool
) -> list[str | None]:
    """Attach a label to every patch, leaving None where OCR failed.

    A fully readable line is not required: whatever is read is applied to the
    matching patches.
    """
    if score_line:
        # A score line reads more accurately in digits-only mode. The decimal
        # point is known from the shape, so the digits are assigned in order.
        in_order = _labels_by_ocr(crop, pieces, cfg)
        if in_order is not None:
            return list(in_order)

    whitelist = "0123456789." if score_line else "0123456789" + LETTERS
    chars = ocr_chars(crop, cfg.threshold, whitelist)

    mask = ink(crop, cfg.threshold)
    used_rows = np.flatnonzero(mask.any(axis=1))
    line_height = int(used_rows[-1] - used_rows[0] + 1) if used_rows.size else crop.shape[0]

    # The decimal point rule only applies where digits were actually read;
        # otherwise every speck on the game screen would become a decimal point.
    read_digits = sum(1 for ch, _, _ in chars if ch.isdigit())
    allow_dot = score_line and read_digits >= 2

    out: list[str | None] = []
    for piece in pieces:
        a, _, b, _ = piece.rect if piece.rect else (0, 0, 0, 0)
        width = max(1, b - a)

        # OCR often misses the decimal point, but its shape gives it away: short
            # and narrow. This applies to score lines only.
        height = piece.mask.shape[0]
        if (
            allow_dot
            and height <= max(2, line_height // 3)
            and piece.mask.shape[1] <= max(2, line_height // 2)
        ):
            out.append("dot")
            continue

        best_overlap, picked_char = 0.0, None
        for ch, cx0, cx1 in chars:
            overlap = min(b, cx1) - max(a, cx0)
            if overlap > best_overlap:
                best_overlap, picked_char = overlap, ch
        if picked_char is None or best_overlap < width * 0.5:
            out.append(None)
        elif picked_char.isdigit():
            out.append(picked_char)
        elif picked_char == ".":
            out.append("dot")
        else:
            out.append("other")  # a letter means this is not a digit
    return out


def ocr_digits(crop: np.ndarray, threshold: int, scale: int = 4) -> str | None:
    """Read the digits at this rectangle with the slower OCR path, or None.

    Labelling is done once and may take its time, so accuracy is preferred.
    """
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        return None

    char = (ink(crop, threshold) * 255).astype(np.uint8)
    char = np.pad(char, 10, constant_values=0)
    image = Image.fromarray(255 - char)  # black on white reads better
    image = image.resize((image.width * scale, image.height * scale), Image.LANCZOS)
    try:
        reading = pytesseract.image_to_string(
            image, config="--psm 13 -c tessedit_char_whitelist=0123456789."
        )
    except Exception:
        return None
    digits = "".join(ch for ch in reading if ch.isdigit())
    return digits or None


def _labels_by_ocr(
    crop: np.ndarray, pieces: list, cfg: Config
) -> list[str] | None:
    """Assign the OCR digits to the patches in order, or None on a count mismatch.

    Which patch is the decimal point is decided by shape; the rest are digits.
    """
    digits = ocr_digits(crop, cfg.threshold)
    if digits is None:
        return None

    mask = ink(crop, cfg.threshold)
    used_rows = np.flatnonzero(mask.any(axis=1))
    if used_rows.size == 0:
        return None
    line_height = int(used_rows[-1] - used_rows[0] + 1)

    kind: list[str] = []
    for x0, x1 in glyph_spans(mask):
        piece = mask[:, x0:x1]
        rows = np.flatnonzero(piece.any(axis=1))
        height = int(rows[-1] - rows[0] + 1) if rows.size else 0
        width = x1 - x0
        low = height <= max(2, line_height // 3) and width <= max(2, line_height // 2)
        kind.append("dot" if low else "digit")

    if len(kind) != len(pieces):
        return None
    if kind.count("digit") != len(digits):
        return None

    out: list[str] = []
    digits_left = list(digits)
    for kind in kind:
        out.append("dot" if kind == "dot" else digits_left.pop(0))
    return out


def read_truth(path: Path) -> dict[str, list[tuple[float, float]]]:
    """Load the hand written truth file.

    One line per screenshot holds the eight slot values; a covered slot is "-".
    """
    out: dict[str, list[tuple[float, float]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        piece = line.split()
        if len(piece) < 2 or line.strip().startswith("#"):
            continue
        values: list[tuple[float, float]] = []
        for cell in piece[1:]:
            if cell == "-":
                values.append(None)  # type: ignore[arg-type]
                continue
            ts, ks = cell.split("/")
            values.append((float(ts), float(ks)))
        out[piece[0]] = values
    return out


def _text_for(value: tuple[float, float]) -> str:
    """Turn (10.5, 8.5) into '10.58.5', the order printed on screen."""
    ts, ks = value
    return f"{ts}{ks}"


def _read_frame(
    frame: np.ndarray, cfg: Config, reader
) -> list[tuple[float, float] | None] | None:
    """Read the whole frame and trust the values only when the rules hold.

    Using the raw reading as a label would bake misreadings into the dataset, so
    all eight slots are validated together.
    """
    if reader is None:
        return None
    from scan import read_slots

    got = read_slots(frame, cfg, reader)
    if not got.complete:
        return None
    readings = [r for r in got.readings if r is not None]
    try:
        interpret(readings)
    except ScanError:
        return None
    return [(r.ts, r.ks) for r in readings]


def samples_from(
    frame: np.ndarray,
    cfg: Config,
    shot: str,
    reader=None,
    truth: list[tuple[float, float] | None] | None = None,
    other_cap: int = 40,
    use_ocr: bool = True,
    digit_like: float = 0.80,
) -> list[Sample]:
    """Extract every patch from one screenshot.

    Labels come from the truth file when given, then from a frame that passed the
    rule check, then from OCR.
    """
    from recognize import similarity

    height, width = frame.shape[:2]
    values = truth if truth is not None else _read_frame(frame, cfg, reader)
    out: list[Sample] = []
    others: list[Sample] = []
    for slot, box in enumerate(cfg.slots):
        normal, raised = slot_windows(cfg, width, height, box)
        places = [("down", normal)] + ([("up", raised)] if raised is not None else [])
        for where, rect in places:
            crop = slot_crop(frame, cfg, rect)
            if crop.size == 0:
                continue
            pieces = glyphs_in(crop, cfg.threshold)
            if not pieces:
                continue

            score_line = is_score_window(crop, cfg.threshold)
            labels: list[str | None] | None = None
            if score_line and values is not None and slot < len(values) and values[slot]:
                char = _text_for(values[slot])
                if len(char) == len(pieces):
                    labels = ["dot" if ch == "." else ch for ch in char]
            if labels is None and use_ocr:
                labels = labels_by_ocr(crop, pieces, cfg, score_line)

            scale = reference_scale(cfg, width, height)
            win_x0, win_y0 = rect[0], rect[1]
            for i, piece in enumerate(pieces):
                # glyphs_in returns coordinates inside the crop, which was scaled to the
                # reference size, so the scale is undone and the window origin added to
                # find the place on the original frame.
                a, b, c, d = piece.rect if piece.rect else (0, 0, 0, 0)
                x0 = win_x0 + round(a / scale)
                y0 = win_y0 + round(b / scale)
                x1 = win_x0 + round(c / scale)
                y1 = win_y0 + round(d / scale)
                form = canonical(piece.gray, piece.mask_wide)
                context = _around(frame, (x0, y0, x1, y1), margin=40)
                label = labels[i] if labels is not None else None
                if label is None:
                    # only patches with no OCR and no truth go to the operator
                    out.append(Sample(form, "unlabeled", shot, slot, where, x0, x1, context))
                elif label == "other":
                    others.append(Sample(form, "other", shot, slot, where, x0, x1, context))
                else:
                    out.append(Sample(form, label, shot, slot, where, x0, x1, context))

    # Digits also appear outside score lines: team numbers, and letters such
        # as O, I or S in team names. Filing those under other would make one
        # shape both a digit and not a digit, which contradicts itself during
        # training, so digit-like patches go to the operator instead.
    digit_forms = [s.form for s in out if s.label.isdigit()]

    similar, rest = [], []
    for s in others:
        if any(similarity(s.form, f) >= digit_like for f in digit_forms):
            similar.append(Sample(s.form, "unlabeled", s.shot, s.slot, s.where, s.x0, s.x1, s.context))
        else:
            rest.append(s)

    out.extend(similar)
    out.extend(_thin_out(rest, other_cap))
    return out


def _around(
    frame: np.ndarray, rect: tuple[int, int, int, int], margin: int = 40
) -> np.ndarray | None:
    """Crop the surroundings so the patch can be located on the original frame.

    The rectangle is outlined in red; the patch alone would not show where it sat.
    """
    x0, y0, x1, y1 = rect
    height, width = frame.shape[:2]
    left, top = max(0, x0 - margin), max(0, y0 - margin // 3)
    right, bottom = min(width, x1 + margin), min(height, y1 + margin // 3)
    piece = frame[top:bottom, left:right].copy()
    if not piece.size:
        return None

    a, b = x0 - left, y0 - top
    c, d = min(piece.shape[1] - 1, x1 - left), min(piece.shape[0] - 1, y1 - top)
    red = np.array([255, 60, 60], dtype=piece.dtype)
    piece[b : d + 1, a] = red
    piece[b : d + 1, c] = red
    piece[b, a : c + 1] = red
    piece[d, a : c + 1] = red
    return piece


def _thin_out(samples: list[Sample], cap: int) -> list[Sample]:
    """Thin out the non-digit patches.

    The game screen and the team names yield far more patches than the digits do.
    Keeping them all would skew the dataset towards "not a digit".
    """
    from recognize import similarity

    kept: list[Sample] = []
    for s in samples:
        if any(similarity(s.form, k.form) >= 0.97 for k in kept):
            continue
        kept.append(s)
        if len(kept) >= cap:
            break
    return kept


def save(samples: list[Sample], out: Path) -> dict[str, int]:
    """Save the patches into per-label folders and record their origin in index.csv."""
    from PIL import Image

    out.mkdir(parents=True, exist_ok=True)
    counted: dict[str, int] = {}
    rows = []
    for i, s in enumerate(samples):
        folder = out / s.label
        folder.mkdir(parents=True, exist_ok=True)
        name = f"{Path(s.shot).stem}_{s.slot + 1}_{s.where}_{i:04d}.png"
        Image.fromarray(np.clip(s.form * 255, 0, 255).astype(np.uint8)).save(folder / name)
        if s.context is not None:
            context_dir = out / "context"
            context_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(s.context).save(context_dir / name)
        counted[s.label] = counted.get(s.label, 0) + 1
        rows.append([s.label, name, s.shot, s.slot + 1, s.where, s.x0, s.x1])

    index = out / "index.csv"
    is_new = not index.exists()
    with index.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["label", "file", "shot", "slot", "where", "x0", "x1"])
        writer.writerows(rows)
    return counted


def main() -> None:
    parser = argparse.ArgumentParser(description="스크린샷에서 학습 자료 만들기")
    parser.add_argument("--config", default=CONFIG_PATH, help="이 스크린샷들에 맞는 설정")
    parser.add_argument("--shots", required=True, help="스크린샷이 든 폴더")
    parser.add_argument("--out", default=DATASET_DIR, help="자료를 쌓을 폴더")
    parser.add_argument("--truth", default=None, help="화면에 적힌 값을 손으로 적어 둔 파일")
    parser.add_argument(
        "--no-ocr", action="store_true", help="레이블 지정에 Tesseract를 사용하지 않음"
    )
    parser.add_argument(
        "--check", action="store_true", help="첫 스크린샷의 OCR 판별 결과만 출력하고 종료"
    )
    parser.add_argument(
        "--tesseract", default=None, help="Tesseract 실행 파일 경로. 자동 탐색에 실패한 경우에만 지정"
    )
    parser.add_argument(
        "--other-cap", type=int, default=OTHER_CAP, help="스크린샷 한 장에서 가져올 '기타' 조각 수"
    )
    parser.add_argument(
        "--digit-like",
        type=float,
        default=DIGIT_LIKE,
        help="이 점수 이상으로 숫자와 유사한 조각은 other 대신 수동 확인으로 분류",
    )
    args = parser.parse_args()

    from PIL import Image

    cfg = Config.load(args.config)
    available, reason = ocr_ready(args.tesseract)
    if args.no_ocr:
        print("OCR 없이 실행합니다. 레이블은 --truth 또는 글자 그림으로만 지정됩니다.")
    elif available:
        print(f"OCR을 쓴다: {reason}")
    else:
        print(f"OCR을 사용할 수 없습니다: {reason}")
        print("이 상태로 진행하면 대부분의 조각이 unlabeled로 분류됩니다.")

    reader = None
    if Path(MODEL_PATH).exists():
        from digits import DigitModel

        reader = DigitModel.load(MODEL_PATH)
        print(f"학습된 모델로도 판독합니다: {MODEL_PATH}")
    else:
        print("학습된 모델이 없습니다. 레이블은 OCR과 --truth로만 지정됩니다.")

    shots = sorted(
        p for p in Path(args.shots).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
    )
    if not shots:
        raise SystemExit(f"스크린샷을 찾을 수 없습니다: {args.shots}")

    truths = read_truth(Path(args.truth)) if args.truth else {}
    if args.check:
        image = np.array(Image.open(shots[0]).convert("RGB"))
        print(f"\n{shots[0].name}의 칸마다 OCR이 읽은 것")
        height, width = image.shape[:2]
        for i, box in enumerate(cfg.slots):
            normal, raised = slot_windows(cfg, width, height, box)
            for where, rect in (("down", normal), ("up", raised)):
                if rect is None:
                    continue
                crop = slot_crop(image, cfg, rect)
                digits = ocr_digits(crop, cfg.threshold)
                chars = ocr_chars(crop, cfg.threshold, "0123456789" + LETTERS)
                print(
                    f"  {i + 1}번 {where}: 숫자만 {digits!r}, "
                    f"글자와 자리 {[(c, round(a)) for c, a, _ in chars]}"
                )
        return

    total: dict[str, int] = {}
    for path in shots:
        frame = np.array(Image.open(path).convert("RGB"))
        samples = samples_from(
            frame, cfg, path.name, reader, truths.get(path.name), args.other_cap,
            not args.no_ocr, args.digit_like,
        )
        counted = save(samples, Path(args.out))
        for k, v in counted.items():
            total[k] = total.get(k, 0) + v
        print(f"{path.name}: " + ", ".join(f"{k} {v}개" for k, v in sorted(counted.items())))

    print("\n모두 합쳐서")
    for k, v in sorted(total.items()):
        print(f"  {k}: {v}개")
    digits = sum(v for k, v in total.items() if k.isdigit() or k == "dot")
    others = total.get("other", 0)
    if digits:
        print(f"  숫자 대 기타 = 1 : {others / digits:.1f}")
    if not digits and not args.no_ocr:
        print(
            "\n레이블이 하나도 지정되지 않았습니다. --check로 OCR 판별 결과를 먼저 확인해 주세요."
        )
    left = total.get("unlabeled", 0)
    if left:
        print(f"\nunlabeled {left}개는 수동 확인이 필요합니다. labeling.py로 레이블을 지정해 주세요.")


if __name__ == "__main__":
    main()
