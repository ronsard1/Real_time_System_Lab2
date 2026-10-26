"""
partition.py - partitioned multiprocessor scheduling (Mini Project 2.3, Section 2.8).

Every heuristic has the same signature  heuristic(tasks, m) -> PartitionResult.
Admission predicate on a core: the per-core Liu & Layland RM bound recomputed for the
task count the core would have after the placement,
        load + u_i <= rm_bound(n_on_core + 1)
never a flat 0.69 or 1.0 threshold.

Remaining capacity of a core (used by Best-Fit and Worst-Fit) is measured the same way,
after the hypothetical placement: rm_bound(n + 1) - (load + u_i).
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from common.rts_sim import Task, rm_bound  # noqa: E402,F401


@dataclass
class Core:
    tasks: list = field(default_factory=list)

    @property
    def load(self):
        return sum(t.U for t in self.tasks)

    def fits(self, t) -> bool:
        return self.load + t.U <= rm_bound(len(self.tasks) + 1) + 1e-12

    def slack_after(self, t) -> float:
        return rm_bound(len(self.tasks) + 1) - (self.load + t.U)


@dataclass
class PartitionResult:
    heuristic: str
    m: int
    cores: list
    unplaced: list

    @property
    def success(self):
        return not self.unplaced

    @property
    def cores_used(self):
        return sum(1 for c in self.cores if c.tasks)

    @property
    def placed_utilization(self):
        return sum(c.load for c in self.cores)

    def mapping(self):
        return {t.name: i for i, c in enumerate(self.cores) for t in c.tasks}


def _run(name, ordered, m, choose):
    cores = [Core() for _ in range(m)]
    unplaced = []
    for t in ordered:
        cand = [c for c in cores if c.fits(t)]
        if not cand:
            unplaced.append(t)
            continue
        choose(cand, t).tasks.append(t)
    return PartitionResult(name, m, cores, unplaced)


def _decreasing(tasks):
    idx = {t.name: i for i, t in enumerate(tasks)}
    return sorted(tasks, key=lambda t: (-t.U, idx[t.name]))


def first_fit(tasks, m):
    """Input order, first core (lowest index) that admits the task."""
    return _run("FF", list(tasks), m, lambda cand, t: cand[0])


def first_fit_decreasing(tasks, m):
    """Decreasing u_i, then First-Fit."""
    return _run("FFD", _decreasing(tasks), m, lambda cand, t: cand[0])


def best_fit_decreasing(tasks, m):
    """Decreasing u_i, admissible core with the least remaining capacity after placement."""
    return _run("BFD", _decreasing(tasks), m, lambda cand, t: min(cand, key=lambda c: c.slack_after(t)))


def worst_fit(tasks, m):
    """Input order, admissible core with the most remaining capacity after placement."""
    return _run("WF", list(tasks), m, lambda cand, t: max(cand, key=lambda c: c.slack_after(t)))


HEURISTICS = {"FF": first_fit, "FFD": first_fit_decreasing, "BFD": best_fit_decreasing, "WF": worst_fit}


def min_cores(tasks, heuristic, limit=64):
    """Smallest m for which the heuristic places every task (for the optional lower-bound gap)."""
    for m in range(1, limit + 1):
        if heuristic(tasks, m).success:
            return m
    return None
