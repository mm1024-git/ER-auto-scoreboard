"""Tournament constants and rank/points conversion.

Placement points, from first to eighth place: 10, 7, 5, 4, 3, 2, 1, 0.
"""

from __future__ import annotations

PROTOCOL = 2  # module interface version; mixing different values is unsafe
FILE_SET = "2026-09-06-a"  # release this file belongs to

TEAM_COUNT = 8

PLACEMENT_POINTS: dict[int, int] = {1: 10, 2: 7, 3: 5, 4: 4, 5: 3, 6: 2, 7: 1, 8: 0}

# All values differ, so the reverse mapping is unambiguous.
_POINTS_TO_RANK: dict[int, int] = {v: k for k, v in PLACEMENT_POINTS.items()}
assert len(_POINTS_TO_RANK) == len(PLACEMENT_POINTS)


class ScanError(Exception):
    """Raised when a reading breaks the scoring rules."""


def points_of(rank: int) -> int:
    if rank not in PLACEMENT_POINTS:
        raise ScanError(f"등수 범위를 벗어났습니다: {rank}")
    return PLACEMENT_POINTS[rank]


def rank_of(points: int) -> int:
    if points not in _POINTS_TO_RANK:
        raise ScanError(f"순위점수 표에 없는 값입니다: {points}")
    return _POINTS_TO_RANK[points]
