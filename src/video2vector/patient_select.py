from __future__ import annotations

from collections import defaultdict

from .detect_track import Detection


class PatientSelector:
    """Choose which tracked person is the subject; everyone else is dropped."""

    def __init__(self, mode: str = "longest_track", min_track_seconds: float = 1.0):
        self.mode = mode
        self.min_track_seconds = min_track_seconds
        self._durations: dict[int, float] = defaultdict(float)
        self._areas: dict[int, float] = defaultdict(float)
        self._counts: dict[int, int] = defaultdict(int)
        self._last_t: dict[int, float] = {}
        self.patient_id: int | None = None
        self.locked = False

    def observe(self, t: float, people: list[Detection]) -> None:
        for p in people:
            if p.track_id is None:
                continue
            tid = p.track_id
            if tid in self._last_t:
                self._durations[tid] += max(0.0, t - self._last_t[tid])
            self._last_t[tid] = t
            self._areas[tid] += p.area
            self._counts[tid] += 1

    def lock(self) -> int | None:
        candidates = {
            tid: dur
            for tid, dur in self._durations.items()
            if dur >= self.min_track_seconds or self._counts[tid] >= 3
        }
        if not candidates:
            candidates = dict(self._durations)
        if not candidates:
            self.patient_id = None
            self.locked = True
            return None
        if self.mode == "largest_box":
            self.patient_id = max(
                candidates,
                key=lambda tid: self._areas[tid] / max(self._counts[tid], 1),
            )
        else:
            # longest_track (default) and enrollment_first_face fall back here
            # until an enrollment embedding is supplied.
            self.patient_id = max(candidates, key=lambda tid: candidates[tid])
        self.locked = True
        return self.patient_id

    def choose(self, people: list[Detection]) -> tuple[Detection | None, list[Detection]]:
        """Return (patient, others). Animals are never patients."""
        if not people:
            return None, []
        if self.patient_id is None:
            # Before lock: prefer the largest person box as a provisional subject.
            provisional = max(people, key=lambda p: p.area)
            others = [p for p in people if p is not provisional]
            return provisional, others
        patient = None
        others = []
        for p in people:
            if p.track_id == self.patient_id:
                patient = p
            else:
                others.append(p)
        return patient, others
