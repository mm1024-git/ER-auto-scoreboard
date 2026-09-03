"""Turn the TS/KS readings into ranks and accumulate the standings.

Two calculations carry the module. Placement points are TS minus KS, and the
teams holding the highest value are still alive. Everything else is validation
around those two facts.
"""

from __future__ import annotations

PROTOCOL = 2  # module interface version; mixing different values is unsafe
FILE_SET = "2026-09-06-a"  # release this file belongs to

from dataclasses import dataclass, field, replace

from rules import PLACEMENT_POINTS, TEAM_COUNT, ScanError, points_of, rank_of


@dataclass(frozen=True)
class SlotReading:
    """One reading taken from a scoreboard slot."""

    ts: float
    ks: float
    raised: bool = False  # True when the card is raised because the team is observed


@dataclass(frozen=True)
class Snapshot:
    """Interpretation of a single scan."""

    ks: tuple[float, ...]
    rank: tuple[int | None, ...]  # None means the team is still alive
    provisional_rank: int  # rank the living teams would take if wiped now
    observed_slot: int | None = None
    ts: tuple[float, ...] = ()

    @property
    def alive(self) -> tuple[int, ...]:
        return tuple(i for i, r in enumerate(self.rank) if r is None)

    @property
    def finished(self) -> bool:
        return len(self.alive) <= 1

    def points(self, slot: int) -> int:
        rank = self.rank[slot]
        return points_of(rank if rank is not None else self.provisional_rank)


def interpret(readings: list[SlotReading]) -> Snapshot:
    """Turn readings into a Snapshot.

    Args:
        readings: one reading per slot.

    Returns:
        The interpreted Snapshot.

    Raises:
        ScanError: the readings break the scoring rules.
    """
    if len(readings) != TEAM_COUNT:
        raise ScanError(f"칸 수가 {TEAM_COUNT}개가 아닙니다: {len(readings)}")

    placements: list[int] = []
    for slot, r in enumerate(readings):
        raw = r.ts - r.ks
        rounded = round(raw)
        if abs(raw - rounded) > 1e-6:
            raise ScanError(f"{slot + 1}번 팀의 순위점수가 정수가 아닙니다: {raw}")
        placements.append(int(rounded))

    top = max(placements)
    alive = [i for i, p in enumerate(placements) if p == top]
    provisional = rank_of(top)
    if provisional != len(alive):
        raise ScanError(
            f"생존 팀은 {len(alive)}팀인데 잠정 순위점수 {top}점은 "
            f"{provisional}등에 해당합니다"
        )

    # Living teams share the provisional rank, so their placement points must
    # be equal and must match the survivor count. One misread slot is caught
    # here.
    expected_alive = points_of(len(alive))
    if top != expected_alive:
        raise ScanError(
            f"{len(alive)}팀 생존 시 순위점수는 {expected_alive}점이어야 하는데 "
            f"{top}점으로 판독되었습니다"
        )

    rank: list[int | None] = [None] * TEAM_COUNT
    used: set[int] = set()
    for slot, p in enumerate(placements):
        if p == top:
            continue
        r = rank_of(p)
        if r <= provisional:
            raise ScanError(f"{slot + 1}번 팀의 등수 {r}이 생존 팀 수보다 앞섭니다")
        if r in used:
            raise ScanError(f"등수 {r}이 두 팀에 중복됩니다")
        used.add(r)
        rank[slot] = r

    expected = set(range(provisional + 1, TEAM_COUNT + 1))
    if used != expected:
        missing = sorted(expected - used)
        raise ScanError(f"탈락 팀 등수가 비어 있습니다: {missing}")

    if len(alive) == 1:
        rank[alive[0]] = 1

    observed = next((i for i, r in enumerate(readings) if r.raised), None)
    return Snapshot(
        ks=tuple(r.ks for r in readings),
        rank=tuple(rank),
        provisional_rank=provisional,
        observed_slot=observed,
        ts=tuple(r.ts for r in readings),
    )


def suspect_slots(readings: list[SlotReading]) -> list[int]:
    """Find which slot, when corrected, would make the frame consistent.

    With a single misreading, restoring that slot makes the rest fit. Exactly one
    candidate identifies the culprit; several leave it undecidable.
    """
    if len(readings) != TEAM_COUNT:
        return []
    try:
        interpret(readings)
        return []
    except ScanError:
        pass

    found: list[int] = []
    for slot in range(TEAM_COUNT):
        for points in PLACEMENT_POINTS.values():
            fixed = list(readings)
            r = fixed[slot]
            fixed[slot] = replace(r, ts=r.ks + points)
            try:
                interpret(fixed)
            except ScanError:
                continue
            found.append(slot)
            break
    return found


@dataclass
class ObserveResult:
    snapshot: Snapshot | None
    committed: bool
    warnings: list[str] = field(default_factory=list)


class RoundTracker:
    """Validate the scans of one round and commit them.

    A value is accepted only after it has been read twice in a row, which filters
    out transition screens and one-off misreadings.
    """

    def __init__(self, confirm_repeats: int = 2) -> None:
        self.confirm_repeats = confirm_repeats
        self.committed: Snapshot | None = None
        self.last_readings: list[SlotReading] | None = None  # used to fill covered slots
        self._pending: list[SlotReading] | None = None
        self._repeats = 0

    def observe(self, readings: list[SlotReading]) -> ObserveResult:
        try:
            snapshot = interpret(readings)
        except ScanError as e:
            self._pending, self._repeats = None, 0
            suspects = suspect_slots(readings)
            text = [str(e)]
            if len(suspects) == 1:
                text.append(f"{suspects[0] + 1}번 팀만 수정하면 규칙에 맞습니다. 해당 팀의 판독 오류로 보입니다")
            return ObserveResult(None, False, text)

        if self._pending == readings:
            self._repeats += 1
        else:
            self._pending, self._repeats = readings, 1
        if self._repeats < self.confirm_repeats:
            return ObserveResult(snapshot, False, [])

        warnings = self._check_monotonic(snapshot)
        if warnings:
            return ObserveResult(snapshot, False, warnings)

        self.committed = snapshot
        self.last_readings = list(readings)
        return ObserveResult(snapshot, True, [])

    def _check_monotonic(self, new: Snapshot) -> list[str]:
        old = self.committed
        if old is None:
            return []
        problems: list[str] = []
        for slot in range(TEAM_COUNT):
            if old.ts and new.ts and new.ts[slot] < old.ts[slot] - 1e-6:
                problems.append(
                    f"{slot + 1}번 팀의 TS가 {old.ts[slot]}에서 "
                    f"{new.ts[slot]}로 감소했습니다"
                )
            if new.ks[slot] < old.ks[slot] - 1e-6:
                problems.append(
                    f"{slot + 1}번 팀의 KS가 {old.ks[slot]}에서 "
                    f"{new.ks[slot]}로 감소했습니다"
                )
            if new.points(slot) < old.points(slot):
                problems.append(
                    f"{slot + 1}번 팀의 순위점수가 {old.points(slot)}에서 "
                    f"{new.points(slot)}로 감소했습니다"
                )
            if old.rank[slot] is not None and new.rank[slot] != old.rank[slot]:
                problems.append(
                    f"{slot + 1}번 팀의 확정 등수가 {old.rank[slot]}에서 "
                    f"{new.rank[slot]}로 변경되었습니다"
                )
        return problems

    def force(self, snapshot: Snapshot) -> None:
        """Store a value edited by hand, without validation."""
        self.committed = snapshot
        self._pending, self._repeats = None, 0


@dataclass(frozen=True)
class RoundResult:
    """Committed record of one finished round."""

    ks: tuple[float, ...]
    rank: tuple[int, ...]

    @staticmethod
    def from_snapshot(s: Snapshot) -> "RoundResult":
        if not s.finished:
            raise ScanError("생존 팀이 남아 있어 라운드를 확정할 수 없습니다")
        return RoundResult(ks=s.ks, rank=tuple(r or 1 for r in s.rank))


@dataclass
class TeamStanding:
    slot: int
    name: str
    total: float
    kill_score: float
    place_score: int
    penalty: float
    last_rank: int | None


class MatchState:
    """Accumulate the rounds. Penalties stay apart and are added to the totals only."""

    def __init__(self, team_names: list[str] | None = None) -> None:
        self.team_names = team_names or [f"팀 {i + 1}" for i in range(TEAM_COUNT)]
        self.rounds: list[RoundResult] = []
        self.penalties: list[dict[int, float]] = []
        self.current = RoundTracker()
        # earlier results carried as one row instead of per round; None means empty
        self.offset: dict[int, dict[str, float | None]] = {}
        # number attached to each committed round; may start above 1
        self.round_numbers: list[int] = []
        self.next_round_no = 1

    def new_match(self) -> None:
        """Clear every record, keeping the team names."""
        self.rounds.clear()
        self.penalties.clear()
        self.offset.clear()
        self.round_numbers.clear()
        self.next_round_no = 1
        self.current = RoundTracker()

    def discard_round(self) -> None:
        """Drop the readings of the round in progress; committed rounds stay."""
        self.current = RoundTracker()

    def set_offset(self, slot: int, ts: float | None, ks: float | None) -> None:
        """Set the offset of one team; missing values overwrite as empty."""
        self.offset[slot] = {"ts": ts, "ks": ks}

    def duplicate_ranks(self, round_index: int) -> set[int]:
        """Return the ranks that occur more than once in that round."""
        if not 0 <= round_index < len(self.rounds):
            return set()
        seen: dict[int, int] = {}
        for rank in self.rounds[round_index].rank:
            if rank is None:
                continue
            seen[rank] = seen.get(rank, 0) + 1
        return {rank for rank, count in seen.items() if count > 1}

    def clear_cell(
        self, round_index: int | None, slot: int, field: str, prune: bool = True
    ) -> None:
        """Clear one cell; a round index of None refers to the offset row."""
        if round_index is None:
            values = self.offset.setdefault(slot, {"ts": None, "ks": None})
            values["ts" if field == "place" else field] = None
            return
        if not 0 <= round_index < len(self.rounds):
            raise ScanError(f"{round_index + 1}라운드가 존재하지 않습니다")
        record = self.rounds[round_index]
        if field == "ks":
            values = list(record.ks)
            values[slot] = None
            self.rounds[round_index] = replace(record, ks=tuple(values))
        elif field == "rank":
            ranks = list(record.rank)
            ranks[slot] = None
            self.rounds[round_index] = replace(record, rank=tuple(ranks))
        elif field == "penalty":
            self.penalties[round_index].pop(slot, None)
        if prune:
            # Clearing the last value drops the round from the list. While a whole
            # row is being cleared the drop happens once at the end, so the
            # remaining cells are not looked up in a round that no longer exists.
            self.drop_empty_rounds()
            self.drop_empty_offset()

    def clear_row(self, round_index: int | None, slot: int) -> None:
        """Clear one team's row; the row stays and only the values are removed."""
        if round_index is None:
            self.offset[slot] = {"ts": None, "ks": None}
            self.drop_empty_offset()
            return
        for name in ("ks", "rank", "penalty"):
            self.clear_cell(round_index, slot, name, prune=False)
        self.drop_empty_rounds()
        self.drop_empty_offset()

    def add_round(
        self,
        ks: list[float | None],
        ranks: list[int | None],
        number: int | None = None,
    ) -> RoundResult:
        """Add one round by hand; a partial row is allowed.

        Args:
            ks: kill scores per slot, None where unknown.
            ranks: ranks per slot, None where unknown.
            number: round number to use; defaults to the next one.

        Returns:
            The stored RoundResult.

        Raises:
            ScanError: the number is taken or a rank is out of range.
        """
        number = self.next_round_no if number is None else number
        if number in self.round_numbers:
            raise ScanError(f"{number}라운드는 이미 존재합니다")
        given = [r for r in ranks if r is not None]
        if any(not 1 <= r <= TEAM_COUNT for r in given):
            raise ScanError(f"등수는 1에서 {TEAM_COUNT} 사이여야 합니다")
        record = RoundResult(ks=tuple(ks), rank=tuple(ranks))
        self.rounds.append(record)
        self.round_numbers.append(number)
        self.penalties.append({})
        if number >= self.next_round_no:
            self.next_round_no = number + 1
        self.drop_empty_rounds()
        return record

    def is_empty_round(self, index: int) -> bool:
        """Return True when every value of that round is empty."""
        record = self.rounds[index]
        if any(v is not None for v in record.ks):
            return False
        if any(v is not None for v in record.rank):
            return False
        return not (
            self.penalties[index] if index < len(self.penalties) else {}
        )

    def drop_empty_offset(self) -> None:
        """Drop the offset row once all eight teams are empty."""
        if self.offset and all(
            values.get("ts") is None and values.get("ks") is None
            for values in self.offset.values()
        ):
            self.offset.clear()

    def drop_empty_rounds(self) -> None:
        """Drop the rounds whose values are all empty."""
        keep = [i for i in range(len(self.rounds)) if not self.is_empty_round(i)]
        self.rounds = [self.rounds[i] for i in keep]
        self.round_numbers = [self.round_numbers[i] for i in keep]
        self.penalties = [
            self.penalties[i] if i < len(self.penalties) else {} for i in keep
        ]

    def number_of(self, index: int) -> int:
        """Return the number attached to the round at that index."""
        if index < len(self.round_numbers):
            return self.round_numbers[index]
        return index + 1

    def finish_round(self, order: list[int] | None = None) -> RoundResult:
        """Commit the current round.

        Args:
            order: ranking order for the slots still alive, best first. The last
                wipe is sometimes missing from the broadcast, so the operator can
                supply it.

        Returns:
            The committed RoundResult.

        Raises:
            ScanError: nothing is committed yet or the number is taken.
        """
        s = self.current.committed
        if s is None:
            raise ScanError("확정된 판독 결과가 없습니다")
        if order is not None:
            s = finish_by_hand(s, order)
        if self.next_round_no in self.round_numbers:
            raise ScanError(f"{self.next_round_no}라운드는 이미 존재합니다")
        result = RoundResult.from_snapshot(s)
        self.rounds.append(result)
        self.round_numbers.append(self.next_round_no)
        self.next_round_no += 1
        self.penalties.append({})
        self.current = RoundTracker()
        return result

    def edit_round(
        self,
        round_index: int,
        slot: int,
        ks: float | None = None,
        rank: int | None = None,
    ) -> None:
        """Edit a committed round by hand; penalties are separate.

        This undoes a slot that the reader committed wrongly.
        """
        if not 0 <= round_index < len(self.rounds):
            raise ScanError(f"{round_index + 1}라운드가 존재하지 않습니다")
        record = self.rounds[round_index]

        if ks is not None:
            value = list(record.ks)
            value[slot] = float(ks)
            record = replace(record, ks=tuple(value))

        if rank is not None:
            if not 1 <= rank <= TEAM_COUNT:
                raise ScanError(f"등수는 1에서 {TEAM_COUNT} 사이여야 합니다: {rank}")
            # Only this cell is written. Duplicates are shown in red and left to the
            # operator; swapping automatically would change untouched cells.
            ranks = list(record.rank)
            ranks[slot] = rank
            record = replace(record, rank=tuple(ranks))

        self.rounds[round_index] = record

    def set_penalty(self, round_index: int, slot: int, value: float) -> None:
        """Set a penalty. Round indexes start at 0; the round in progress is len(rounds)."""
        while len(self.penalties) <= round_index:
            self.penalties.append({})
        self.penalties[round_index][slot] = value

    def standings(self, include_current: bool = True) -> list[TeamStanding]:
        kill = [0.0] * TEAM_COUNT
        place = [0] * TEAM_COUNT
        penalty = [0.0] * TEAM_COUNT
        last_rank: list[int | None] = [None] * TEAM_COUNT

        for slot, values in self.offset.items():
            kill[slot] += values.get("ks") or 0.0
            # The offset TS is everything the team scored so far. Any KS given is
            # already part of it, so only the placement share is added here.
            ts = values.get("ts")
            if ts is not None:
                place[slot] += ts - (values.get("ks") or 0.0)

        for i, r in enumerate(self.rounds):
            for slot in range(TEAM_COUNT):
                if r.ks[slot] is not None:
                    kill[slot] += r.ks[slot]
                if r.rank[slot] is not None:
                    place[slot] += points_of(r.rank[slot])
                    last_rank[slot] = r.rank[slot]
                penalty[slot] += self.penalties[i].get(slot, 0.0)

        cur = self.current.committed
        if include_current and cur is not None:
            pen = (
                self.penalties[len(self.rounds)]
                if len(self.penalties) > len(self.rounds)
                else {}
            )
            for slot in range(TEAM_COUNT):
                kill[slot] += cur.ks[slot]
                place[slot] += cur.points(slot)
                penalty[slot] += pen.get(slot, 0.0)
                if cur.rank[slot] is not None:
                    last_rank[slot] = cur.rank[slot]

        rows = [
            TeamStanding(
                slot=slot,
                name=self.team_names[slot],
                total=kill[slot] + place[slot] + penalty[slot],
                kill_score=kill[slot],
                place_score=place[slot],
                penalty=penalty[slot],
                last_rank=last_rank[slot],
            )
            for slot in range(TEAM_COUNT)
        ]
        # Ties break on kill score, then on the rank of the most recent round;
        # teams with no rank yet go last.
        rows.sort(
            key=lambda t: (
                -t.total,
                -t.kill_score,
                t.last_rank if t.last_rank is not None else TEAM_COUNT + 1,
            )
        )
        return rows

    def to_dict(self) -> dict:
        cur = self.current.committed
        return {
            "round": len(self.rounds) + 1,
            "alive": list(cur.alive) if cur else [],
            "observed_slot": cur.observed_slot if cur else None,
            "standings": [
                {
                    "place": i + 1,
                    "slot": t.slot,
                    "name": t.name,
                    "total": t.total,
                    "ks": t.kill_score,
                    "ps": t.place_score,
                    "penalty": t.penalty,
                    "last_rank": t.last_rank,
                }
                for i, t in enumerate(self.standings())
            ],
        }


def finish_by_hand(s: Snapshot, order: list[int]) -> Snapshot:
    """Assign ranks to the living slots in the order the operator supplied.

    As many ranks remain as there are living teams, and the slot listed first
    takes the best of them.
    """
    alive = list(s.alive)
    if sorted(order) != sorted(alive):
        raise ScanError(
            f"생존 팀은 {[i + 1 for i in alive]}인데 "
            f"{[i + 1 for i in order]}를 입력받았습니다"
        )
    ranks = list(s.rank)
    for place, slot in enumerate(order):
        ranks[slot] = place + 1
    return replace(s, rank=tuple(ranks))


def snapshot_with_rank(s: Snapshot, slot: int, rank: int | None) -> Snapshot:
    """Return a Snapshot with one slot's rank replaced, for manual edits."""
    ranks = list(s.rank)
    ranks[slot] = rank
    return replace(s, rank=tuple(ranks))
