"""Train a small classifier on the labelled dataset.

    python train.py --data clean --out digits.npz

The input is one 11x18 patch and the output is one of twelve classes: digits 0
to 9, the decimal point and "other". The decimal point is decided by position
when reading, but it stays a class so the model never reports it as a digit.

Only numpy is used: 198 inputs and twelve classes need a single hidden layer.
"""

from __future__ import annotations

FILE_SET = "2026-09-04-g"  # release this file belongs to

import argparse
from pathlib import Path

import numpy as np

from settings import (
    CLEAN_DIR,
    DECAY,
    EPOCHS,
    HIDDEN,
    MODEL_PATH,
    SEED,
    SMOOTHING,
    TARGET,
)

LABELS = [str(i) for i in range(10)] + ["dot", "other"]
WIDTH, HEIGHT = 11, 18


def load_folder(folder: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load patches from the label folders.

    Args:
        folder: dataset root holding one folder per class.

    Returns:
        A tuple of (patches, labels, source screenshot names).
    """
    from PIL import Image

    images: list[np.ndarray] = []
    labels: list[int] = []
    shots: list[str] = []
    for index, label in enumerate(LABELS):
        for path in sorted((folder / label).glob("*.png")):
            picture = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
            if picture.shape != (HEIGHT, WIDTH):
                continue
            images.append(picture)
            labels.append(index)
            # file names look like <screenshot>_<slot>_<position>_<index>.png
            shots.append(path.name.rsplit("_", 3)[0])
    if not images:
        raise SystemExit(f"{folder}에서 읽을 조각을 찾을 수 없습니다")
    return np.stack(images), np.array(labels), shots


def split_by_shot(
    x: np.ndarray, y: np.ndarray, shots: list[str], ratio: float = 0.2, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split into train and test sets by screenshot.

    Splitting by patch would put the same glyph from one frame on both sides,
    which measures memorisation and inflates the score.
    """
    names = sorted(set(shots))
    rng = np.random.default_rng(seed)
    rng.shuffle(names)

    if len(names) < 2:
        # With a single screenshot a per-frame split is impossible; a per-patch
        # split is used instead and the reported score is optimistic.
        print("스크린샷이 한 장뿐이어서 조각 단위로 분할합니다. 검증 성적은 실제보다 높게 나옵니다.")
        order = rng.permutation(len(y))
        held = np.zeros(len(y), dtype=bool)
        held[order[: max(1, int(len(y) * ratio))]] = True
        return x[~held], y[~held], x[held], y[held]

    held_names = set(names[: max(1, int(len(names) * ratio))])
    held = np.array([s in held_names for s in shots])
    if held.all() or not held.any():  # an empty side would defeat the split
        held = np.array([s == names[0] for s in shots])
    return x[~held], y[~held], x[held], y[held]


def shake(picture: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Jitter one patch.

    Shifts of one or two pixels, blurring and stroke weight changes cover the
    variation seen when the frame shakes or the encoding quality changes.
    """
    out = np.roll(picture, int(rng.integers(-1, 2)), axis=0)
    out = np.roll(out, int(rng.integers(-1, 2)), axis=1)

    if rng.random() < 0.5:  # blur: mix with the neighbours
        padded = np.pad(out, 1, mode="edge")
        total = np.zeros_like(out)
        for a in (0, 1, 2):
            for b in (0, 1, 2):
                total += padded[a : a + HEIGHT, b : b + WIDTH]
        out = 0.5 * out + 0.5 * (total / 9.0)

    if rng.random() < 0.3:  # weight: bend the brightness to thicken or thin the strokes
        out = np.clip(out ** rng.uniform(0.6, 1.6), 0.0, 1.0)
    return out.astype(np.float32)


def augment(
    x: np.ndarray, y: np.ndarray, target: int, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Grow the small classes until every class holds target samples."""
    rng = np.random.default_rng(seed)
    grown_x, grown_y = [x], [y]
    for label in np.unique(y):
        have = x[y == label]
        missing = target - len(have)
        if missing <= 0:
            continue
        picked = have[rng.integers(0, len(have), missing)]
        made = np.empty_like(picked)
        for i, picture in enumerate(picked):
            made[i] = shake(picture, rng)
        grown_x.append(made)
        grown_y.append(np.full(missing, label))
    return np.concatenate(grown_x), np.concatenate(grown_y)


def class_weights(y: np.ndarray, power: float = 0.5) -> np.ndarray:
    """Return the per-class loss weights.

    Weighting by the plain inverse frequency suppresses the large "other" class
    so far that even training samples stop being classified as other, so the
    square root of the inverse is used.
    """
    counts = np.bincount(y, minlength=len(LABELS)).astype(np.float32)
    weights = np.where(counts > 0, (counts.sum() / np.maximum(counts, 1)) ** power, 0.0)
    seen = counts > 0
    if seen.any():
        weights[seen] /= weights[seen].mean()
    return weights


def train(
    x: np.ndarray,
    y: np.ndarray,
    hidden: int = 96,
    epochs: int = 60,
    rate: float = 0.2,
    decay: float = 1e-4,
    smoothing: float = 0.05,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Train a classifier with a single hidden layer.

    Two regularisers are applied. Weight decay keeps the weights small and label
    smoothing targets 0.95 instead of 1.0; without them the outputs saturate and
    an unfamiliar shape still receives a confident class.
    """
    rng = np.random.default_rng(seed)
    inputs = x.reshape(len(x), -1)

    smooth = smoothing / len(LABELS)
    targets = np.full((len(y), len(LABELS)), smooth, dtype=np.float32)
    targets[np.arange(len(y)), y] = 1.0 - smoothing + smooth

    weights = class_weights(y)[y][:, None]

    w1 = rng.normal(0, np.sqrt(2 / inputs.shape[1]), (inputs.shape[1], hidden)).astype(
        np.float32
    )
    b1 = np.zeros(hidden, dtype=np.float32)
    w2 = rng.normal(0, np.sqrt(2 / hidden), (hidden, len(LABELS))).astype(np.float32)
    b2 = np.zeros(len(LABELS), dtype=np.float32)

    batch = 128
    for _ in range(epochs):
        order = rng.permutation(len(inputs))
        for start in range(0, len(order), batch):
            picked = order[start : start + batch]
            xb, tb, sb = inputs[picked], targets[picked], weights[picked]

            hidden_out = np.maximum(0, xb @ w1 + b1)
            scores = hidden_out @ w2 + b2
            scores -= scores.max(axis=1, keepdims=True)
            probs = np.exp(scores)
            probs /= probs.sum(axis=1, keepdims=True)

            grad = (probs - tb) * sb / len(picked)
            grad_w2 = hidden_out.T @ grad + decay * w2
            grad_b2 = grad.sum(axis=0)
            grad_hidden = (grad @ w2.T) * (hidden_out > 0)
            grad_w1 = xb.T @ grad_hidden + decay * w1
            grad_b1 = grad_hidden.sum(axis=0)

            w1 -= rate * grad_w1
            b1 -= rate * grad_b1
            w2 -= rate * grad_w2
            b2 -= rate * grad_b2
    return {"W1": w1, "b1": b1, "W2": w2, "b2": b2}


def predict(weights: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    """Return per-class probabilities for each patch."""
    inputs = x.reshape(len(x), -1)
    hidden_out = np.maximum(0, inputs @ weights["W1"] + weights["b1"])
    scores = hidden_out @ weights["W2"] + weights["b2"]
    scores -= scores.max(axis=1, keepdims=True)
    probs = np.exp(scores)
    return probs / probs.sum(axis=1, keepdims=True)


def report(weights: dict[str, np.ndarray], x: np.ndarray, y: np.ndarray) -> None:
    """Print per-class accuracy and the most common confusion."""
    answer = predict(weights, x).argmax(axis=1)
    print("\n클래스별 성적")
    for index, label in enumerate(LABELS):
        here = y == index
        count = int(here.sum())
        if count == 0:
            continue
        right = int((answer[here] == index).sum())
        wrong = answer[here][answer[here] != index]
        note = ""
        if len(wrong):
            counted = np.bincount(wrong, minlength=len(LABELS))
            worst = int(counted.argmax())
            note = f"  주로 {LABELS[worst]}로 틀림({int(counted[worst])}개)"
        print(f"  {label:>5}: {right}/{count} ({right / count:.1%}){note}")
    print(f"  전체 : {(answer == y).mean():.1%}")


def sweep(weights: dict[str, np.ndarray], x: np.ndarray, y: np.ndarray) -> None:
    """Measure wrong and missed rates over a grid of acceptance thresholds.

    Only digit classes are counted. A wrong value is worse than a missing one, so
    pick the row with no wrong readings and the lowest missed rate.
    """
    probs = predict(weights, x)
    order = np.argsort(probs, axis=1)[:, ::-1]
    best = order[:, 0]
    gap = probs[np.arange(len(y)), best] - probs[np.arange(len(y)), order[:, 1]]
    top = probs[np.arange(len(y)), best]

    digits = np.array([LABELS[i].isdigit() for i in range(len(LABELS))])
    here = digits[y]
    if not here.any():
        return

    print("\n판별 기준별 성적 (숫자 클래스만)")
    print("  확률  차이   오판별      미판별")
    for min_score in (0.5, 0.6, 0.7, 0.8, 0.9):
        for min_margin in (0.1, 0.2, 0.3, 0.5):
            accepted = (top >= min_score) & (gap >= min_margin)
            reading = here & accepted
            wrong = int((reading & (best != y)).sum())
            missed = int((here & ~accepted).sum())
            count = int(here.sum())
            print(
                f"  {min_score:.1f}  {min_margin:.1f}   "
                f"{wrong}/{count} ({wrong / count:.1%})  {missed}/{count} ({missed / count:.1%})"
            )
    print("  선택한 값을 settings.py의 MODEL_MIN_SCORE와 MODEL_MIN_MARGIN에 입력해 주세요.")
    print("  app 폴더와 training 폴더의 settings.py에 같은 값을 넣어야 합니다.")


def suspicious_others(
    weights: dict[str, np.ndarray],
    folder: Path,
    threshold: float = 0.9,
    margin: float = 0.5,
) -> list[tuple[str, str, float, float]]:
    """List patches labelled other that the model reads confidently as digits.

    The top probability alone also catches shapes that resemble nothing, so the
    gap to the runner-up is required as well.
    """
    from PIL import Image

    images: list[np.ndarray] = []
    kept: list[Path] = []
    for path in sorted((folder / "other").glob("*.png")):
        picture = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
        if picture.shape == (HEIGHT, WIDTH):
            images.append(picture)
            kept.append(path)
    if not images:
        return []

    found: list[tuple[str, str, float, float]] = []
    for path, probs in zip(kept, predict(weights, np.stack(images))):
        order = np.argsort(probs)[::-1]
        best, second = int(order[0]), int(order[1])
        gap = float(probs[best] - probs[second])
        if LABELS[best].isdigit() and probs[best] >= threshold and gap >= margin:
            found.append((path.name, LABELS[best], float(probs[best]), gap))
    return sorted(found, key=lambda item: -item[3])


def main() -> None:
    parser = argparse.ArgumentParser(description="확인이 끝난 자료로 분류기 학습")
    parser.add_argument("--data", default=CLEAN_DIR, help="레이블 지정이 끝난 자료 폴더")
    parser.add_argument("--out", default=MODEL_PATH, help="학습 가중치를 저장할 파일")
    parser.add_argument("--hidden", type=int, default=HIDDEN, help="은닉층 크기")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="학습 반복 횟수")
    parser.add_argument("--target", type=int, default=TARGET, help="클래스별 증강 목표 샘플 수")
    parser.add_argument("--decay", type=float, default=DECAY, help="가중치 감쇠")
    parser.add_argument("--smoothing", type=float, default=SMOOTHING, help="라벨 스무딩")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    folder = Path(args.data)
    x, y, shots = load_folder(folder)
    counts = np.bincount(y, minlength=len(LABELS))
    print("데이터 분포")
    for index, label in enumerate(LABELS):
        if counts[index]:
            print(f"  {label:>5}: {counts[index]}개")
    print(f"  합계 : {len(y)}개, 스크린샷 {len(set(shots))}장")

    train_x, train_y, test_x, test_y = split_by_shot(x, y, shots, seed=args.seed)
    print(f"\n학습 {len(train_y)}개, 검증 {len(test_y)}개 (스크린샷 단위로 분할)")

    grown_x, grown_y = augment(train_x, train_y, args.target, seed=args.seed)
    print(f"증강 후 학습 데이터 {len(grown_y)}개")

    weights = train(
        grown_x,
        grown_y,
        args.hidden,
        args.epochs,
        decay=args.decay,
        smoothing=args.smoothing,
        seed=args.seed,
    )
    report(weights, test_x, test_y)
    sweep(weights, test_x, test_y)

    np.savez(args.out, **weights)
    size = Path(args.out).stat().st_size / 1024
    print(f"\n가중치를 {args.out}에 저장했습니다 ({size:.0f}KB)")

    found = suspicious_others(weights, folder)
    if found:
        print(f"\nother로 분류되었으나 숫자로 판별되는 조각 {len(found)}개")
        for name, label, best, gap in found[:20]:
            print(f"  {name}: {label}일 확률 {best:.0%}, 2등과 차이 {gap:.0%}")
        print("  잘못 지정된 항목은 해당 숫자 폴더로 옮긴 뒤 다시 학습해 주세요.")


if __name__ == "__main__":
    main()
