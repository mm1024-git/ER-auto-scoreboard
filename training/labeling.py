"""Label the dataset patches produced by dataset.py.

    python labeling.py --dataset dataset --clean clean

Patches with the same shape are grouped, and the operator answers once per
group: a digit, a decimal point, or "not a digit". Each question also shows the
surrounding area, since only that tells whether the patch came from a score line
or from the background.

Labelled patches move to the clean folder, so whatever stays in the dataset
folder is still unlabelled and a later run asks only about those.

Place this file in: training/
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from config import Config
from settings import CLEAN_DIR, CONFIG_PATH, DATASET_DIR
from recognize import (
    EDGE_MARGIN,
    to_gray,
    crop_ink,
    glyph_spans,
    ink,
    line_spans,
    similarity,
)
from scan import detect_observed, is_score_window, reference_scale, slot_crop, slot_windows


@dataclass
class Piece:
    """One cut glyph and the place it came from."""

    mask: np.ndarray
    rect: tuple[int, int, int, int] | None = None  # rectangle in frame coordinates
    context: np.ndarray | None = None  # original image including the surroundings
    path: Path | None = None  # source file when it came from a dataset folder
    label: str | None = None  # label already attached in the dataset folder
    gray: np.ndarray | None = None  # raw brightness image
    mask_wide: np.ndarray | None = None  # ink mask with the same width as gray

    @property
    def form(self) -> np.ndarray:
        """Shape used for comparison and storage; brightness is preferred."""
        from recognize import TEMPLATE_HEIGHT, TEMPLATE_WIDTH, canonical

        if self.gray is not None and self.gray.shape == (TEMPLATE_HEIGHT, TEMPLATE_WIDTH):
            return self.gray.astype(np.float32)  # dataset patches are already normalised
        if self.gray is None:
            return canonical(self.mask)
        return canonical(self.gray, self.mask_wide)


def glyphs_in(
    img: np.ndarray, threshold: int = 190, origin: tuple[int, int] = (0, 0)
) -> list[Piece]:
    """Cut every glyph out of one image patch."""
    mask = ink(img, threshold)
    gray = to_gray(img)
    ox, oy = origin
    out: list[Piece] = []
    for y0, y1 in line_spans(mask):
        row, bright = mask[y0:y1], gray[y0:y1]
        for x0, x1 in glyph_spans(row):
            patch = row[:, x0:x1]
            if np.count_nonzero(patch) >= 2:  # the decimal point counts as a glyph
                # widen by one pixel on both sides to keep faint stroke edges
                a = max(0, x0 - EDGE_MARGIN)
                b = min(row.shape[1], x1 + EDGE_MARGIN)
                out.append(
                    Piece(
                        crop_ink(patch),
                        (ox + x0, oy + y0, ox + x1, oy + y1),
                        gray=bright[:, a:b],
                        mask_wide=row[:, a:b],
                    )
                )
    return out


def glyphs_in_frame(
    frame: np.ndarray, cfg: Config, reader=None
) -> list[Piece]:
    """Scan the configured slots, or the lower band when none are configured."""
    height, width = frame.shape[:2]
    if cfg.slots:
        # As when reading: the observed slot is detected first and only one
        # rectangle is scanned per slot. Scanning both would also collect the
        # team number and name left at the lowered position.
        observed = detect_observed(frame, cfg, reader)
        out: list[Piece] = []
        scale = reference_scale(cfg, width, height)

        def piece(rect):
            x0, y0, x1, y1 = rect
            crop = slot_crop(frame, cfg, rect)
            pieces = glyphs_in(crop, cfg.threshold)
            for piece in pieces:  # map the rectangle back to original image coordinates
                a, b, c, d = piece.rect
                piece.rect = (
                    x0 + round(a / scale), y0 + round(b / scale),
                    x0 + round(c / scale), y0 + round(d / scale),
                )
            return crop, pieces

        for i, box in enumerate(cfg.slots):
            normal, raised = slot_windows(cfg, width, height, box)
            places = [("내려온 자리", normal)]
            if raised is not None:
                places.append(("올라간 자리", raised))

            usable = []
            for where, rect in places:
                crop, pieces = piece(rect)
                if is_score_window(crop, cfg.threshold):
                    usable.append((where, rect, pieces))

            if not usable:
                # Neither rectangle looks like a score line. Since the right one cannot
                # be decided yet, both are asked; later answers settle it.
                for where, rect in places:
                    _, pieces = piece(rect)
                    print(f"  {i + 1}번 칸 {where} {rect} 에서 {len(pieces)}조각 — 판단 보류, 양쪽 모두 질의")
                    out.extend(pieces)
                continue

            for where, rect, pieces in usable:
                print(f"  {i + 1}번 칸 {where} {rect} 에서 {len(pieces)}조각")
                out.extend(pieces)
        if observed is None and cfg.raise_ratio > 0:
            print("  올라간 칸을 판별하지 못했습니다. 이 화면에서는 해당 칸을 건너뜁니다.")
        return out

    ax0, ay0, ax1, ay1 = cfg.area(width, height)
    top = ay0 + int((ay1 - ay0) * 0.7)
    return glyphs_in(frame[top:ay1, ax0:ax1], cfg.threshold, (ax0, top))


@dataclass
class Cluster:
    rep: Piece
    members: list[Piece] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.members)


def cluster_glyphs(pieces: list[Piece], threshold: float = 0.90) -> list[Cluster]:
    """Group patches by overlapping shape, most frequent group first.

    One digit can split into two groups by brightness or position; giving both
    the same answer costs one extra question and no accuracy.
    """
    clusters: list[Cluster] = []
    for piece in pieces:
        for c in clusters:
            if similarity(piece.form, c.rep.form) >= threshold:
                c.members.append(piece)
                break
        else:
            clusters.append(Cluster(rep=piece, members=[piece]))
    clusters.sort(key=lambda c: -c.count)
    return clusters


def ascii_art(mask: np.ndarray) -> str:
    return "\n".join("".join("#" if v else "." for v in row) for row in mask)


def already_known(
    patch: np.ndarray,
    reader,
    min_score: float = 0.70,
    min_margin: float = 0.10,
) -> str | None:
    """Report whether a group can be skipped, using the reading thresholds.

    A stricter bar here would ask again about shapes that already read fine.
    """
    if reader is None:
        return None
    ch, score, other = reader.match(patch, already=True)
    return ch if score >= min_score and score - other >= min_margin else None


def current_guess(patch: np.ndarray, reader) -> str:
    """Return the text describing how this shape currently reads."""
    if reader is None:
        return ""
    ch, score, other = reader.match(patch, already=True)
    if ch == "?" or score <= 0:
        return "현재 판별되지 않는 형태입니다."
    return f"지금은 {ch}으로 읽힌다 (점수 {score:.2f}, 다음 후보와 차이 {score - other:.2f})."


def context_image(piece: Piece, frame: np.ndarray | None, margin: int = 60):
    """Return the surrounding image and the rectangle of the patch inside it."""
    if piece.context is not None:
        return piece.context, None
    if frame is None or piece.rect is None:
        return None, None
    x0, y0, x1, y1 = piece.rect
    height, width = frame.shape[:2]
    left, top = max(0, x0 - margin), max(0, y0 - margin // 4)
    right, bottom = min(width, x1 + margin), min(height, y1 + margin // 4)
    return frame[top:bottom, left:right], (x0 - left, y0 - top, x1 - left, y1 - top)


def needs_every_cluster(reader) -> bool:
    """Report whether the table still lacks any digit or the decimal point.

    Scores cannot be trusted before the table is complete: without an 8 an 8
    reads as 0 and nothing detects it.
    """
    if reader is None:
        return True
    # the decimal point is decided by position, so only digits are counted
    return bool(set("0123456789") - reader.chars)


class PictureLabeler:
    """Show the cut shape next to the original frame and read the answer.

    The upper image is the surrounding area with the patch outlined; the lower
    one is the normalised shape used for comparison.
    """

    def __init__(
        self,
        clusters: list[Cluster],
        reader,
        frame: np.ndarray | None,
    ) -> None:
        import tkinter as tk

        self.frame = frame
        self.reader = reader
        self.out = None
        self.ask_all = needs_every_cluster(reader)
        self.pending = [
            c
            for c in clusters
            if self.ask_all or already_known(c.rep.form, reader) is None
        ]
        self.확인중: str | None = None
        self.index = 0
        self.answered = 0

        self.root = tk.Tk()
        self.root.title("이 모양이 무슨 글자인가")
        self.info = tk.Label(self.root, anchor="w", justify="left")
        self.info.pack(fill="x")
        self.canvas = tk.Canvas(self.root, width=820, height=380, background="#111111")
        self.canvas.pack()

        bar = tk.Frame(self.root)
        bar.pack(fill="x")
        self.bar = bar
        tk.Label(bar, text="글자:").pack(side="left")
        self.entry = tk.Entry(bar, width=4)
        self.entry.pack(side="left")
        self.entry.focus_set()
        tk.Button(bar, text="넣기", command=self.submit).pack(side="left")
        tk.Button(bar, text="건너뛰기", command=self.skip).pack(side="left")
        tk.Button(bar, text="그만", command=self.root.destroy).pack(side="right")
        self.root.bind("<Return>", lambda e: self.submit())
        self.root.bind("<Escape>", lambda e: self.root.destroy())
        self.show()

    def show(self) -> None:
        """Draw the current group, reporting the reason instead of leaving it blank."""
        try:
            self._draw()
        except Exception as e:  # a blank window would hide the cause
            import traceback

            traceback.print_exc()
            self.canvas.delete("all")
            self.canvas.create_text(
                10, 10, anchor="nw", fill="#ff8888", text=f"그리지 못했다: {e}"
            )

    def _draw(self) -> None:
        from PIL import Image, ImageTk

        self.canvas.delete("all")
        # Groups that now read thanks to earlier answers are skipped; before the
        # table is complete every group is asked.
        while (
            not self.ask_all
            and self.index < len(self.pending)
            and already_known(self.pending[self.index].rep.form, self.out)
        ):
            self.index += 1
        if self.index >= len(self.pending):
            self.info.config(text=f"모든 묶음을 처리하였습니다. {self.answered}개를 지정하였습니다. 그만을 클릭해 주세요.")
            return

        cluster = self.pending[self.index]
        context, spot = context_image(cluster.rep, self.frame)
        guess = current_guess(cluster.rep.form, self.reader)
        self.info.config(
            text=f"[{self.index + 1}/{len(self.pending)}] 이 모양이 {cluster.count}번 나왔다. "
            "위가 원본 화면, 아래가 대조에 쓰는 모양이다.\n"
            + (self.note(cluster) or guess)
        )

        if context is not None and context.size:
            # Scale from both width and height so the image fits; scaling on one
            # axis alone would cut off the bottom.
            width_room, height_room = 800, 230
            scale = max(
                1,
                min(8, width_room // max(1, context.shape[1]), height_room // max(1, context.shape[0])),
            )
            view = Image.fromarray(context).resize(
                (context.shape[1] * scale, context.shape[0] * scale), Image.NEAREST
            )
            self.context_photo = ImageTk.PhotoImage(view)
            self.canvas.create_image(10, 10, anchor="nw", image=self.context_photo)
            if spot is not None:
                x0, y0, x1, y1 = (v * scale for v in spot)
                self.canvas.create_rectangle(
                    10 + x0, 10 + y0, 10 + x1, 10 + y1, outline="#ff4444", width=2
                )
            bottom = 20 + view.height
        else:
            self.canvas.create_text(
                10,
                10,
                anchor="nw",
                fill="#ffcc66",
                text=(
                    "이 조각의 주변 그림이 없습니다. dataset/context 폴더가 비어 있거나\n"
                    "이전 판으로 생성한 자료입니다. dataset을 다시 생성하면 함께 저장됩니다."
                ),
            )
            bottom = 60

        mask = cluster.rep.mask
        scale = max(4, min(8, 120 // max(1, mask.shape[0])))
        shape = Image.fromarray((mask * 255).astype(np.uint8)).convert("RGB")
        shape = shape.resize((mask.shape[1] * scale, mask.shape[0] * scale), Image.NEAREST)
        self.mask_photo = ImageTk.PhotoImage(shape)
        self.canvas.create_image(10, bottom, anchor="nw", image=self.mask_photo)
        # outline the shape so a dark patch is visible on the dark background
        self.canvas.create_rectangle(
            9,
            bottom - 1,
            10 + shape.width,
            bottom + shape.height,
            outline="#666666",
        )

    def note(self, cluster: Cluster) -> str:
        """Extra text per group; dataset mode shows the label already attached."""
        return ""

    def submit(self) -> None:
        if self.index >= len(self.pending):
            return
        got = self.entry.get().strip()
        self.entry.delete(0, "end")
        if not got:
            return self.skip()
        if got not in "0123456789.x":
            self.info.config(text="입력 가능한 값은 0에서 9, 소수점 ., 숫자 아님 x 입니다")
            return

        form = self.pending[self.index].rep.form
        self.out.add(got, form, already=True)
        self.answered += 1
        self.index += 1
        self.확인중 = None
        self.show()

    def skip(self) -> None:
        self.index += 1
        self.entry.delete(0, "end")
        self.show()

    def run(self):
        self.root.mainloop()
        return self.out


DATASET_LABELS = set("0123456789") | {"dot", "other"}


def to_label(answer: str) -> str | None:
    """Convert a typed character to a label name, or None when unusable.

    The decimal point cannot be a folder name and becomes "dot".
    """
    label = "dot" if answer == "." else ("other" if answer in ("x", "X") else answer)
    return label if label in DATASET_LABELS else None


def cluster_label(c: Cluster) -> str | None:
    """Return the label already attached to a group, by majority."""
    counts: dict[str, int] = {}
    for piece in c.members:
        if piece.label:
            counts[piece.label] = counts.get(piece.label, 0) + 1
    if not counts:
        return None
    return max(counts, key=lambda k: counts[k])


def label_dataset(
    clusters: list[Cluster],
    folder: Path,
    cfg: Config,
    text_mode: bool = False,
    clean: Path | None = None,
) -> int:
    """Ask once per group and move every file of that group.

    The label attached by OCR is shown for confirmation.
    """
    labeled = 0
    if not text_mode:
        try:
            return DatasetLabeler(clusters, folder, clean).run()
        except Exception as e:
            print(f"창을 띄우지 못해 글자판으로 묻는다: {e}")

    for i, c in enumerate(clusters):
        label = cluster_label(c)
        print(f"\n[{i + 1}/{len(clusters)}] 이 형태가 {c.count}회 등장했습니다.")
        if label:
            print(f"지금은 {label}으로 붙어 있다. 맞으면 엔터.")
        print(ascii_art(c.rep.form > 0.5))
        answer = input(
            "무슨 글자인가 (숫자는 0에서 9, 소수점은 ., 숫자가 아니면 x, "
            "넘기려면 엔터): "
        ).strip()
        if not answer:
            continue  # Enter skips, so a stray key press loses nothing
        label = to_label(answer)
        if label is None:
            print("쓰지 않는 글자다. 넣을 수 있는 것은 0123456789 . 또는 x")
            continue
        labeled += move_labeled(c.members, label, folder, clean)
    return labeled


class DatasetLabeler(PictureLabeler):
    """Window for labelling a dataset folder; an answer moves the whole group."""

    def __init__(
        self, clusters: list[Cluster], folder: Path, clean: Path | None = None
    ) -> None:
        self.folder = folder
        self.clean = clean
        self.moved = 0
        super().__init__(clusters, None, None)
        self.root.title("레이블 붙이기")

        import tkinter as tk

        tk.Button(self.bar, text="숫자 아님(other)", command=self.mark_other).pack(side="left")
        tk.Button(self.bar, text="건너뛰기(엔터)", command=self.skip).pack(side="left")
        self.root.bind("<Escape>", lambda e: self.root.destroy())

    def mark_other(self) -> None:
        """Send this group to the non-digit class."""
        self.entry.delete(0, "end")
        self.entry.insert(0, "x")
        self.submit()

    def note(self, cluster: Cluster) -> str:
        label = cluster_label(cluster)
        prefix = (
            f"OCR 판별 결과: {label}." if label else "OCR이 판별하지 못했습니다."
        )
        return (
            prefix
            + " 숫자면 0에서 9, 소수점은 ., 숫자가 아니면 x, 넘기려면 엔터."
            + " 레이블을 지정한 조각만 clean 폴더로 이동합니다."
        )

    def submit(self) -> None:
        if self.index >= len(self.pending):
            return
        answer = self.entry.get().strip()
        self.entry.delete(0, "end")
        if not answer:
            return self.skip()  # Enter skips, so a stray key press loses nothing
        label = to_label(answer)
        if label is None:
            self.info.config(text="입력 가능한 값은 0에서 9, 소수점 ., 숫자 아님 x 입니다")
            return
        self.moved += move_labeled(
            self.pending[self.index].members, label, self.folder, self.clean
        )
        self.answered += 1
        self.index += 1
        self.show()

    def run(self) -> int:
        self.root.mainloop()
        return self.moved


def _check_file_set() -> None:
    """Stop when the local modules come from different releases."""
    from recognize import check_file_set

    check_file_set()


def pieces_from_dataset(folder: Path, labels: list[str] | None = None) -> list[Piece]:
    """Load patches from a dataset folder built by dataset.py.

    Without labels every labelled folder is read as well, and the existing label
    is carried on the patch.
    """
    from PIL import Image

    folders = labels or (sorted("0123456789") + ["dot", "other", "unlabeled"])
    out: list[Piece] = []
    for label_name in folders:
        for path in sorted((folder / label_name).glob("*.png")):
            form = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
            context = folder / "context" / path.name
            context = np.array(Image.open(context).convert("RGB")) if context.exists() else None
            out.append(
                Piece(
                    form > 0.5,
                    None,
                    context,
                    gray=form,
                    path=path,
                    label=None if label_name == "unlabeled" else label_name,
                )
            )
    return out


def move_labeled(
    pieces: list[Piece], label: str, folder: Path, clean: Path | None = None
) -> int:
    """Move the labelled files and return how many were moved.

    With clean given the files go there, so whatever stays in folder is still
    unlabelled and a later run asks only about those. The context image follows.
    """
    import csv

    target_dir = clean or folder
    moved = []
    (target_dir / label).mkdir(parents=True, exist_ok=True)
    for piece in pieces:
        if piece.path is None or not piece.path.exists():
            continue
        piece.path.replace(target_dir / label / piece.path.name)
        moved.append(piece.path.name)

        context = folder / "context" / piece.path.name
        if context.exists():
            (target_dir / "context").mkdir(parents=True, exist_ok=True)
            context.replace(target_dir / "context" / piece.path.name)

    if moved:
        record = target_dir / "labels.csv"
        first = not record.exists()
        with record.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if first:
                writer.writerow(["label", "file"])
            writer.writerows([[label, name] for name in moved])
    return len(moved)


def pieces_from_dir(directory: Path, threshold: int) -> list[Piece]:
    """Load the unread patches together with their stored context images."""
    from PIL import Image

    out: list[Piece] = []
    for path in sorted(directory.glob("glyph_*.png")):
        if path.stem.endswith("_around"):
            continue
        mask = crop_ink(ink(np.array(Image.open(path).convert("RGB")), threshold))
        around = path.with_name(f"{path.stem}_around.png")
        context = np.array(Image.open(around).convert("RGB")) if around.exists() else None
        out.append(Piece(mask, None, context))
    return out


def main() -> None:
    _check_file_set()

    parser = argparse.ArgumentParser(description="데이터셋 레이블 지정")
    parser.add_argument(
        "--dataset", default=DATASET_DIR, help="dataset.py가 생성한 자료 폴더"
    )
    parser.add_argument(
        "--clean", default=CLEAN_DIR, help="레이블을 지정한 조각을 옮길 폴더"
    )
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--text", action="store_true", help="창 대신 콘솔로 질의")
    args = parser.parse_args()

    path = Path(args.config)
    cfg = Config.load(path) if path.exists() else Config()

    folder = Path(args.dataset)
    pieces = pieces_from_dataset(folder)
    if not pieces:
        print(f"{folder}에 레이블을 지정할 조각이 없습니다")
        return

    missing = sum(1 for piece in pieces if piece.context is None)
    if missing:
        print(
            f"주변 그림이 없는 조각이 {missing}/{len(pieces)}개입니다. "
            "해당 묶음은 조각만 보고 판단해야 합니다.\n"
            "이전 판으로 생성한 자료인 경우 dataset 폴더를 삭제한 뒤 다시 생성해 주세요."
        )

    clusters = cluster_glyphs(pieces)
    clean_dir = Path(args.clean)
    print(f"조각 {len(pieces)}개를 형태 {len(clusters)}종으로 묶었습니다")
    print(
        f"레이블을 지정한 조각은 {clean_dir}으로 이동합니다. "
        f"{folder}에 남은 조각이 미처리 대상입니다."
    )
    labeled = label_dataset(clusters, folder, cfg, args.text, clean_dir)
    print(f"조각 {labeled}개를 이동했습니다")


if __name__ == "__main__":
    main()
