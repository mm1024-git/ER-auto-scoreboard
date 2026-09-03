"""Read one glyph with the trained weights.

The reader interface (match, complete, sparse) is kept stable so that calling
code does not need to know how a glyph is recognised.

    reader, min_score, min_margin = load_reader(cfg)
"""

from __future__ import annotations

PROTOCOL = 2  # module interface version; mixing different values is unsafe
FILE_SET = "2026-09-06-a"  # release this file belongs to

from pathlib import Path

import numpy as np

from recognize import TEMPLATE_HEIGHT, TEMPLATE_WIDTH, canonical

LABELS = [str(i) for i in range(10)] + ["dot", "other"]

# The model returns probabilities, so the thresholds differ from template
# matching. Values measured on the data can be set in the config file.
from settings import MODEL_MIN_MARGIN, MODEL_MIN_SCORE, MODEL_PATH


class DigitModel:
    """Classify a single glyph patch.

    Exposes the same methods as the old template table so the reading code does
    not change.
    """

    def __init__(self, weights: dict[str, np.ndarray]) -> None:
        self.weights = {k: np.asarray(v, dtype=np.float32) for k, v in weights.items()}

    @classmethod
    def load(cls, path: str | Path) -> "DigitModel":
        data = np.load(str(path))
        missing = {"W1", "b1", "W2", "b2"} - set(data.files)
        if missing:
            raise ValueError(f"가중치 파일에 없는 항목이 있다: {sorted(missing)}")
        return cls({k: data[k] for k in ("W1", "b1", "W2", "b2")})

    def probabilities(self, patch: np.ndarray) -> np.ndarray:
        """Return class probabilities for one patch."""
        flat = patch.reshape(1, -1).astype(np.float32)
        hidden = np.maximum(0, flat @ self.weights["W1"] + self.weights["b1"])
        scores = hidden @ self.weights["W2"] + self.weights["b2"]
        scores -= scores.max(axis=1, keepdims=True)
        probs = np.exp(scores)
        return (probs / probs.sum(axis=1, keepdims=True))[0]

    def match(
        self, patch: np.ndarray, mask: np.ndarray | None = None, already: bool = False
    ) -> tuple[str, float, float]:
        """Return the best character, its probability and the runner-up probability.

                The caller decides acceptance from these three values. A decimal point is
        returned as '.', anything that is not a digit as '?'.
        """
        form = canonical(patch, mask, already)
        if form.shape != (TEMPLATE_HEIGHT, TEMPLATE_WIDTH):
            return "?", 0.0, 0.0
        probs = self.probabilities(form)
        order = np.argsort(probs)[::-1]
        best, second = int(order[0]), int(order[1])
        label = LABELS[best]
        if label == "dot":
            label = "."
        elif label == "other":
            label = "?"
        return label, float(probs[best]), float(probs[second])

    @property
    def chars(self) -> set[str]:
        return set("0123456789.")

    @property
    def complete(self) -> bool:
        return True

    @property
    def sparse(self) -> bool:
        return False

    @property
    def table(self) -> dict[str, list[np.ndarray]]:
        """Always empty; kept so callers can treat this like the old table."""
        return {}


def load_reader(cfg) -> tuple[object, float, float]:
    """Load the trained weights.

    Args:
        cfg: configuration holding the model path and thresholds.

    Returns:
        A tuple of (reader, min_score, min_margin).

    Raises:
        SystemExit: the weights file is missing.
    """
    model_path = Path(getattr(cfg, "model_path", MODEL_PATH))
    if not model_path.exists():
        raise SystemExit(
            f"{model_path} 파일이 없어 숫자를 읽을 수 없습니다.\n"
            "  실행 파일과 같은 폴더에 digits.npz를 넣어 주세요."
        )
    model = DigitModel.load(model_path)
    min_score = MODEL_MIN_SCORE
    min_margin = MODEL_MIN_MARGIN
    print(f"학습한 모델로 읽는다: {model_path}")
    return model, min_score, min_margin
