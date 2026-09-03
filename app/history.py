"""Correct readings with the scoring rules and with their history.

Three layers are applied.

Per frame: the placement pattern of the eight teams is checked and, when it is
broken, the smallest correction that restores it is applied. Lowering TS is
preferred; KS is only touched when nothing else explains the frame.

Small window: a new value is shown at once, and rolled back when it turns out to
have been a single-frame spike.

Big window: only value changes are kept, and the cost of keeping the current
value is weighed against the cost of dropping it.

Both windows keep the raw readings next to the fixed ones, because a rule based
repair is deterministic: storing only fixed values would follow a wrong repair.
"""

from __future__ import annotations

PROTOCOL = 2
FILE_SET = "2026-09-05-a"

from dataclasses import dataclass, replace

from model import SlotReading
from rules import PLACEMENT_POINTS, TEAM_COUNT, points_of

from settings import BIG_WINDOW, REVERT_GAP, SMALL_WINDOW

# Repair preference; lower is tried first. Lowering TS is safest, raising TS
# invents points and is the last resort.
KIND_PENALTY = {"ts_down": 0, "ks": 1, "ts_up": 3}


def placement_of(reading: SlotReading) -> float:
    """Return the placement points of a slot: TS minus KS."""
    return round(reading.ts - reading.ks, 1)


def pattern_ok(places: list[float]) -> bool:
    """Check that the placement pattern is valid.

    Living teams share the provisional rank, so their placement points are equal.
    Sorted ranks hold one repeated rank, as many times as teams are alive,
    followed by the remaining ranks up to eight.
    """
    ranks = []
    for place in places:
        rank = _rank_of_points(place)
        if rank is None:
            return False
        ranks.append(rank)

    top = min(ranks)
    alive = ranks.count(top)
    if alive != top:
        return False
    rest = sorted(r for r in ranks if r != top)
    return rest == list(range(top + 1, TEAM_COUNT + 1))


def _rank_of_points(points: float) -> int | None:
    for rank, value in PLACEMENT_POINTS.items():
        if abs(value - points) < 1e-6:
            return rank
    return None


@dataclass
class Repair:
    """Result of repairing a single frame."""

    readings: list[SlotReading]
    changed: list[str]          # messages to show the operator
    ok: bool                    # whether the frame is valid after repair


def repair_frame(
    readings: list[SlotReading], settled: set[int] | None = None
) -> Repair:
    """Repair a broken placement pattern with the fewest changes.

    Args:
        readings: the eight slot readings.
        settled: slots eliminated long ago, which are never touched.

    Returns:
        A Repair holding the corrected readings and the messages.
    """
    keep = settled or set()
    places = [placement_of(r) for r in readings]
    if pattern_ok(places):
        return Repair(list(readings), [], True)

    values = sorted(PLACEMENT_POINTS.values(), reverse=True)
    best: tuple[int, int, list[SlotReading], list[str]] | None = None

    for cost in (1, 2):
        for slots in _combinations(range(TEAM_COUNT), cost):
            if any(s in keep for s in slots):
                continue
            for kinds in _product(("ts_down", "ks", "ts_up"), cost):
                fixed = list(readings)
                notes: list[str] = []
                rank_penalty = sum(KIND_PENALTY[k] for k in kinds)
                for slot, kind in zip(slots, kinds):
                    r = fixed[slot]
                    for target in values:
                        if kind in ("ts_up", "ts_down"):
                            new_ts = round(r.ks + target, 1)
                            if kind == "ts_down" and new_ts >= r.ts:
                                continue
                            if kind == "ts_up" and new_ts <= r.ts:
                                continue
                            fixed[slot] = replace(r, ts=new_ts)
                        else:
                            new_ks = round(r.ts - target, 1)
                            if new_ks < 0:
                                continue
                            fixed[slot] = replace(r, ks=new_ks)
                        if pattern_ok([placement_of(x) for x in fixed]):
                            break
                    else:
                        break
                    notes.append(
                        f"{slot + 1}번 팀의 {'KS' if kind == 'ks' else 'TS'}를 "
                        f"{fixed[slot].ks if kind == 'ks' else fixed[slot].ts}로 보정"
                    )
                else:
                    if pattern_ok([placement_of(x) for x in fixed]):
                        score = (cost, rank_penalty)
                        if best is None or score < (best[0], best[1]):
                            best = (cost, rank_penalty, fixed, notes)
        if best is not None:
            break

    if best is None:
        return Repair(list(readings), [], False)
    return Repair(best[2], best[3], True)


def _combinations(items, size):
    from itertools import combinations

    return combinations(items, size)


def _product(items, size):
    from itertools import product

    return product(items, repeat=size)


@dataclass
class Entry:
    """One entry of the big window."""

    frame: int      # frame this entry came from; used to locate it
    raw: float      # value as read from the screen
    fixed: float    # value after repair and rollback
    dwell: int = 1  # frames spent on this value


class SmallWindow:
    """Roll back a single-frame spike after the fact."""

    def __init__(self, size: int = SMALL_WINDOW) -> None:
        self.size = size
        self.raw: list[float] = []     # raw readings; never edited, they are the evidence
        self.fixed: list[float] = []   # fixed values; only these are rolled back

    def add(self, raw: float, fixed: float) -> None:
        self.raw.append(raw)
        self.fixed.append(fixed)
        if len(self.raw) > self.size:
            self.raw.pop(0)
            self.fixed.pop(0)

    def rollback(self) -> float | None:
        """Return the value to roll back to, or None.

        A value counts as a spike when it appeared in exactly one frame and the
        next reading returned to the previous value.
        """
        if len(self.fixed) < 3 or len(self.raw) < 2:
            return None

        spike = self.fixed[-1]
        settled = None
        for value in reversed(self.fixed[:-1]):
            if abs(value - spike) > 1e-6:
                settled = value
                break
        if settled is None:
            return None

        # has the newest reading returned to the pre-spike value?
        if abs(self.raw[-1] - settled) > 1e-6:
            return None
        # was it read in exactly one frame? twice means a real change
        if sum(1 for value in self.raw if abs(value - spike) < 1e-6) != 1:
            return None
        return settled


def _rising(values: list[float]) -> bool:
    """Return True when the values never decrease."""
    return all(b >= a - 1e-6 for a, b in zip(values, values[1:]))


class BigWindow:
    """Keep only value changes and judge the trend.

    Raw and fixed values live in the same entry so both lines cover one span.
    """

    def __init__(self, size: int = BIG_WINDOW) -> None:
        self.size = size
        self.entries: list[Entry] = []

    def add(self, frame: int, raw: float, fixed: float) -> None:
        """Append an entry on change; otherwise extend the dwell count."""
        if self.entries and abs(self.entries[-1].raw - raw) < 1e-6 and abs(
            self.entries[-1].fixed - fixed
        ) < 1e-6:
            self.entries[-1].dwell += 1
            return
        self.entries.append(Entry(frame, raw, fixed))
        self._drop_spike()
        if len(self.entries) > self.size:
            self.entries.pop(0)

    def _drop_spike(self) -> None:
        """Drop a value that appeared for one frame and vanished.

        In [0, 8, 0] the middle value is a spike rather than part of the trend.
        """
        if len(self.entries) < 3:
            return
        first, middle, last = self.entries[-3:]
        if middle.dwell != 1:
            return
        if abs(first.fixed - last.fixed) > 1e-6 or abs(first.raw - last.raw) > 1e-6:
            return
        del self.entries[-2]
        self.collapse()

    def edit(self, frame: int, fixed: float, raw: float | None = None) -> bool:
        """Edit the entry from the given frame, located by frame number."""
        for entry in self.entries:
            if entry.frame == frame:
                entry.fixed = fixed
                if raw is not None:
                    entry.raw = raw
                self.collapse()
                return True
        return False

    def collapse(self) -> None:
        """Merge neighbouring entries that now hold the same values."""
        merged: list[Entry] = []
        for entry in self.entries:
            if (
                merged
                and abs(merged[-1].raw - entry.raw) < 1e-6
                and abs(merged[-1].fixed - entry.fixed) < 1e-6
            ):
                merged[-1].dwell += entry.dwell
                continue
            merged.append(entry)
        self.entries = merged

    def revert_to(self) -> float | None:
        """Return the value to revert to, or None.

        Only readings that arrived after the current value was fixed are weighed.
        The cost of keeping the current value is compared with the cost of
        dropping it, and the longer dwell wins when the costs are equal.
        """
        if len(self.entries) < 3:
            return None
        current = self.entries[-1].fixed

        # find where the current value was fixed
        start = 0
        for i in range(len(self.entries) - 1, -1, -1):
            if abs(self.entries[i].fixed - current) > 1e-6:
                start = i + 1
                break
        after = self.entries[start:]
        if len(after) < 2:
            return None

        raws = [e.raw for e in after]
        keep_cost = sum(1 for r in raws if abs(r - current) > 1e-6)
        if keep_cost < 2:
            return None

        other = [r for r in raws if abs(r - current) > 1e-6]
        if not _rising(other):
            return None
        drop_cost = len(raws) - len(other)

        if keep_cost - drop_cost < REVERT_GAP:
            return None
        if keep_cost == drop_cost and after[-1].dwell > 1:
            return None
        return other[-1]


class Timeline:
    """Hold a small and a big window per slot and correct the values.

    The correction can be switched off, since it may add errors of its own.
    """

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled
        self.frame = 0
        self.small: dict[tuple[int, str], SmallWindow] = {}
        self.big: dict[tuple[int, str], BigWindow] = {}

    def _windows(self, key: tuple[int, str]) -> tuple[SmallWindow, BigWindow]:
        if key not in self.small:
            self.small[key] = SmallWindow()
            self.big[key] = BigWindow()
        return self.small[key], self.big[key]

    def feed(
        self, raw: list[SlotReading], fixed: list[SlotReading]
    ) -> tuple[list[SlotReading], list[str]]:
        """Feed one frame and return the corrected readings and messages."""
        if not self.enabled:
            return list(fixed), []

        self.frame += 1
        out = list(fixed)
        notes: list[str] = []

        for slot in range(len(out)):
            for field in ("ks", "ts"):
                small, big = self._windows((slot, field))
                raw_value = getattr(raw[slot], field)
                value = getattr(out[slot], field)

                small.add(raw_value, value)
                back = small.rollback()
                if back is not None:
                    # The applied value was a single-frame spike: roll back the fixed
                    # line of the small window, both lines of the big window and the
                    # current value. The raw line stays as evidence.
                    small.fixed[-1] = back
                    if big.entries:
                        big.edit(big.entries[-1].frame, fixed=back, raw=back)
                    out[slot] = replace(out[slot], **{field: back})
                    value = back
                    notes.append(
                        f"{slot + 1}번 팀의 {field.upper()} 단일 프레임 오류를 {back}로 복원"
                    )

                big.add(self.frame, raw_value, value)
                target = big.revert_to()
                if target is not None:
                    out[slot] = replace(out[slot], **{field: target})
                    big.entries[-1].fixed = target
                    small.fixed[-1] = target
                    notes.append(
                        f"{slot + 1}번 팀의 {field.upper()} 흐름 불일치를 {target}로 복원"
                    )
        return out, notes

    def reset(self) -> None:
        """Forget the history when a new round starts."""
        self.frame = 0
        self.small.clear()
        self.big.clear()
