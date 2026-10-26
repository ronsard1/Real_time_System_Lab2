"""
rts_sim.py - shared real-time scheduling helpers for Unit 2 (Mini Projects 2.1, 2.2, 2.3).

Contents
  Task                      periodic task record (C, T, D, derived U)
  total_utilization, rm_bound, hyperperiod
  rta                       Response Time Analysis (fixed priority), returns R and iterations
  edf_demand_test           exact EDF test for constrained deadlines (processor demand)
  simulate                  tick-by-tick scheduler: 'rm', 'dm', 'edf', 'llf'
  simulate_resources        three-task style inversion scenario, with or without PIP
  plot_gantt / plot_two     Gantt-style charts with matplotlib broken_barh

Everything is a pure-Python discrete simulation in integer time units (1 tick = 1 ms).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import reduce

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# --------------------------------------------------------------------------- tasks


@dataclass
class Task:
    name: str
    C: int
    T: int
    D: int | None = None          # relative deadline, defaults to T (implicit deadline)

    def __post_init__(self):
        if self.D is None:
            self.D = self.T
        if self.C <= 0 or self.T <= 0 or self.D <= 0:
            raise ValueError(f"{self.name}: C, T and D must be positive")

    @property
    def U(self) -> float:
        return self.C / self.T


def total_utilization(tasks) -> float:
    return sum(t.U for t in tasks)


def rm_bound(n: int) -> float:
    """Liu & Layland sufficient bound n(2^(1/n) - 1)."""
    if n <= 0:
        return 1.0
    return n * (2 ** (1.0 / n) - 1)


def hyperperiod(tasks) -> int:
    return reduce(lambda a, b: a * b // math.gcd(a, b), (t.T for t in tasks))


def is_implicit(tasks) -> bool:
    return all(t.D == t.T for t in tasks)


def priority_order(tasks, policy="rm"):
    """Highest priority first. RM: shortest period. DM: shortest relative deadline.
    Ties are broken by input order, which is deterministic and documented."""
    key = (lambda t: t.T) if policy == "rm" else (lambda t: t.D)
    idx = {t.name: i for i, t in enumerate(tasks)}
    return sorted(tasks, key=lambda t: (key(t), idx[t.name]))


# --------------------------------------------------------------------------- analysis


def rta(tasks, policy="rm", max_iter=1000):
    """Response Time Analysis (Section 2.5.2).
    R_i = C_i + sum_{j in hp(i)} ceil(R_i / T_j) C_j, iterated from R = C_i.
    Returns a list of dicts in priority order: name, R (or None if R > D), iterations, ok, trace."""
    order = priority_order(tasks, policy)
    out = []
    for i, ti in enumerate(order):
        hp = order[:i]
        R = ti.C
        trace = [R]
        it = 0
        ok = None
        while it < max_iter:
            it += 1
            new = ti.C + sum(math.ceil(R / tj.T) * tj.C for tj in hp)
            trace.append(new)
            if new == R:
                ok = R <= ti.D
                break
            if new > ti.D:
                R = new
                ok = False
                break
            R = new
        out.append({"name": ti.name, "R": R, "D": ti.D, "iterations": it,
                    "ok": bool(ok), "trace": trace})
    return out


def edf_demand_test(tasks):
    """Exact EDF test for synchronous constrained-deadline tasks (processor demand criterion):
    for every absolute deadline t up to L = H + max D, dbf(t) <= t, where
    dbf(t) = sum max(0, floor((t - D_i)/T_i) + 1) * C_i.  Requires U <= 1 first.
    Returns (ok, first_violation_t or None)."""
    if total_utilization(tasks) > 1 + 1e-12:
        return False, 0
    L = hyperperiod(tasks) + max(t.D for t in tasks)
    deadlines = sorted({k * t.T + t.D for t in tasks for k in range(L // t.T + 1) if k * t.T + t.D <= L})
    for d in deadlines:
        dbf = sum(max(0, (d - t.D) // t.T + 1) * t.C for t in tasks)
        if dbf > d:
            return False, d
    return True, None


# --------------------------------------------------------------------------- simulation


@dataclass
class Job:
    task: Task
    k: int
    release: int
    deadline: int
    remaining: int
    finish: int | None = None
    missed: bool = False

    @property
    def jid(self):
        return (self.task.name, self.k)


@dataclass
class SimResult:
    policy: str
    horizon: int
    timeline: list                 # per tick: task name or None (idle)
    jobs: list = field(default_factory=list)
    switches: int = 0              # number of dispatches of a job different from the previous one
    preemptions: int = 0           # dispatches that interrupted an unfinished job

    @property
    def misses(self):
        return [j for j in self.jobs if j.missed]

    def max_response(self, name):
        rs = [j.finish - j.release for j in self.jobs if j.task.name == name and j.finish is not None]
        return max(rs) if rs else None


def simulate(tasks, sim_time, policy="rm"):
    """Tick-by-tick simulation. A job of task i is released at every multiple of T_i.
    rm  : fixed priority by period          dm : fixed priority by relative deadline
    edf : smallest absolute deadline        llf: smallest laxity (d - t - remaining)
    Ties: RM/DM by priority order, EDF/LLF by priority order of the task (deterministic).
    A job that passes its deadline is marked missed but keeps running (no job is dropped),
    so overload shows up as a cascade of misses rather than a crash."""
    if policy in ("rm", "dm"):
        order = priority_order(tasks, policy)
    else:
        order = priority_order(tasks, "rm")
    prio = {t.name: i for i, t in enumerate(order)}

    ready: list[Job] = []
    jobs: list[Job] = []
    timeline = []
    prev = None
    switches = preempt = 0
    for t in range(sim_time):
        for tk in tasks:
            if t % tk.T == 0:
                j = Job(tk, t // tk.T, t, t + tk.D, tk.C)
                ready.append(j)
                jobs.append(j)
        for j in ready:
            if not j.missed and t >= j.deadline and j.remaining > 0:
                j.missed = True
        if not ready:
            timeline.append(None)
            continue
        if policy in ("rm", "dm"):
            cur = min(ready, key=lambda j: (prio[j.task.name], j.release))
        elif policy == "edf":
            cur = min(ready, key=lambda j: (j.deadline, prio[j.task.name], j.release))
        elif policy == "llf":
            cur = min(ready, key=lambda j: (j.deadline - t - j.remaining, prio[j.task.name], j.release))
        else:
            raise ValueError(policy)
        if prev is None or cur.jid != prev.jid:
            switches += 1
            if prev is not None and prev.remaining > 0:
                preempt += 1
        cur.remaining -= 1
        timeline.append(cur.task.name)
        if cur.remaining == 0:
            cur.finish = t + 1
            if cur.finish > cur.deadline:
                cur.missed = True
            ready.remove(cur)
            prev = cur
        else:
            prev = cur
    # unfinished jobs whose deadline is inside the horizon are misses
    for j in ready:
        if j.deadline <= sim_time:
            j.missed = True
    return SimResult(policy, sim_time, timeline, jobs, switches, preempt)


def segments(timeline):
    """Convert a per-tick timeline into {name: [(start, length), ...]}."""
    segs = {}
    t = 0
    n = len(timeline)
    while t < n:
        name = timeline[t]
        s = t
        while t < n and timeline[t] == name:
            t += 1
        segs.setdefault(name, []).append((s, t - s))
    return segs


# --------------------------------------------------------------------------- priority inversion


@dataclass
class RTask:
    """Task for the resource scenario. segments: list of ('cpu', n) or ('cs', n, resource)."""
    name: str
    prio: int                       # larger number = higher priority
    arrival: int
    segs: list


def simulate_resources(rtasks, protocol="none", horizon=40):
    """Single-CPU, fixed-priority, preemptive simulation with binary semaphores.
    protocol='none' : plain priorities (unbounded inversion possible)
    protocol='pip'  : Priority Inheritance (Section 2.7) - a lock holder runs at the highest
                      priority of the tasks it blocks, restored when it releases the lock.
    Returns per-tick timeline entries (name, state) plus per-task finish and blocked ticks."""
    state = {}
    for rt in rtasks:
        state[rt.name] = {"seg": 0, "left": rt.segs[0][1], "done": False, "finish": None,
                          "holding": None, "blocked_ticks": 0, "eff": rt.prio, "wait_start": None}
    owner = {}                     # resource -> task name
    by = {rt.name: rt for rt in rtasks}
    timeline = []                  # (time, running name or None, {name: 'blocked'|'ready'})
    for t in range(horizon):
        active = [rt for rt in rtasks if t >= rt.arrival and not state[rt.name]["done"]]
        if not active:
            timeline.append((t, None, {}))
            if all(state[r.name]["done"] for r in rtasks):
                break
            continue
        # effective priorities
        for rt in rtasks:
            state[rt.name]["eff"] = rt.prio
        blocked = {}
        for rt in active:
            s = state[rt.name]
            seg = rt.segs[s["seg"]]
            if seg[0] == "cs" and s["holding"] != seg[2]:
                res = seg[2]
                if res in owner and owner[res] != rt.name:
                    blocked[rt.name] = owner[res]
        if protocol == "pip":
            changed = True
            while changed:              # transitive inheritance
                changed = False
                for b, holder in blocked.items():
                    if state[holder]["eff"] < state[b]["eff"]:
                        state[holder]["eff"] = state[b]["eff"]
                        changed = True
        runnable = [rt for rt in active if rt.name not in blocked]
        status = {rt.name: ("blocked" if rt.name in blocked else "ready") for rt in active}
        for b in blocked:
            state[b]["blocked_ticks"] += 1
        if not runnable:
            timeline.append((t, None, status))
            continue
        cur = max(runnable, key=lambda r: (state[r.name]["eff"], r.prio))
        s = state[cur.name]
        seg = cur.segs[s["seg"]]
        if seg[0] == "cs" and s["holding"] is None:
            owner[seg[2]] = cur.name
            s["holding"] = seg[2]
        s["left"] -= 1
        boosted = s["eff"] > cur.prio
        status[cur.name] = "run_cs_boost" if boosted else ("run_cs" if seg[0] == "cs" else "run")
        timeline.append((t, cur.name, status))
        if s["left"] == 0:
            if seg[0] == "cs":
                del owner[seg[2]]
                s["holding"] = None
            s["seg"] += 1
            if s["seg"] >= len(cur.segs):
                s["done"] = True
                s["finish"] = t + 1
            else:
                s["left"] = cur.segs[s["seg"]][1]
    return timeline, state


# --------------------------------------------------------------------------- plotting

PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
           "#e377c2", "#17becf", "#bcbd22", "#7f7f7f", "#393b79", "#ad494a"]


def task_colors(names):
    return {n: PALETTE[i % len(PALETTE)] for i, n in enumerate(names)}


def draw_gantt(ax, tasks, res: SimResult, title, colors=None, show_idle=True):
    names = [t.name for t in tasks]
    colors = colors or task_colors(names)
    segs = segments(res.timeline)
    rows = names + (["idle"] if show_idle else [])
    ypos = {n: len(rows) - 1 - i for i, n in enumerate(rows)}
    for n in names:
        if n in segs:
            ax.broken_barh(segs[n], (ypos[n] - 0.35, 0.7), facecolors=colors[n], edgecolor="black", linewidth=0.4)
    if show_idle and None in segs:
        ax.broken_barh(segs[None], (ypos["idle"] - 0.35, 0.7), facecolors="#d9d9d9",
                       hatch="///", edgecolor="#7f7f7f", linewidth=0.4)
    for tk in tasks:
        for r in range(0, res.horizon + 1, tk.T):
            ax.plot([r, r], [ypos[tk.name] + 0.36, ypos[tk.name] + 0.5], color="black", lw=0.8)
            ax.plot(r, ypos[tk.name] + 0.5, marker="^", color="black", ms=3)
            if tk.D != tk.T and r + tk.D <= res.horizon:
                ax.plot(r + tk.D, ypos[tk.name] + 0.42, marker="v", color="#d62728", ms=4)
    for j in res.misses:
        if j.deadline <= res.horizon:
            ax.plot(j.deadline, ypos[j.task.name], marker="x", color="red", ms=9, mew=2)
    ax.set_yticks(list(ypos.values()))
    ax.set_yticklabels(list(ypos.keys()))
    ax.set_xlim(0, res.horizon)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    step = 1 if res.horizon <= 30 else (2 if res.horizon <= 60 else max(5, res.horizon // 25))
    ax.set_xticks(range(0, res.horizon + 1, step))
    ax.grid(axis="x", alpha=0.35)
    miss_txt = f", {len(res.misses)} deadline miss(es)" if res.misses else ", no deadline misses"
    ax.set_title(title + miss_txt, fontsize=10)


def plot_two(tasks, r_top, r_bot, t_top, t_bot, path, suptitle=None):
    names = [t.name for t in tasks]
    colors = task_colors(names)
    h = max(3.2, 0.42 * (len(tasks) + 1) + 1.3)
    fig, axes = plt.subplots(2, 1, figsize=(11, 2 * h), sharex=True)
    draw_gantt(axes[0], tasks, r_top, t_top, colors)
    draw_gantt(axes[1], tasks, r_bot, t_bot, colors)
    axes[1].set_xlabel("time (ms)")
    handles = [Patch(color=colors[n], label=n) for n in names] + \
        [Patch(facecolor="#d9d9d9", hatch="///", edgecolor="#7f7f7f", label="idle")]
    fig.legend(handles=handles, loc="upper right", ncol=min(len(handles), 6), fontsize=8,
               bbox_to_anchor=(0.99, 1.0))
    if suptitle:
        fig.suptitle(suptitle, x=0.02, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def draw_resource_timeline(ax, rtasks, timeline, title):
    order = sorted(rtasks, key=lambda r: -r.prio)
    ypos = {r.name: len(order) - 1 - i for i, r in enumerate(order)}
    style = {"run": "#2ca02c", "run_cs": "#ff7f0e", "run_cs_boost": "#9467bd", "blocked": "#d62728"}
    for t, cur, status in timeline:
        for name, st in status.items():
            if st == "ready":
                continue
            ax.broken_barh([(t, 1)], (ypos[name] - 0.3, 0.6), facecolors=style[st],
                           edgecolor="black" if st != "blocked" else style[st], linewidth=0.4,
                           alpha=1.0 if st != "blocked" else 0.35)
        # ready but waiting (preempted) shown as a thin line
        for name, st in status.items():
            if st == "ready" and name != cur:
                ax.plot([t, t + 1], [ypos[name], ypos[name]], color="#7f7f7f", lw=1, ls=":")
    for r in rtasks:
        ax.plot(r.arrival, ypos[r.name] + 0.42, marker="^", color="black", ms=5)
    ax.set_yticks(list(ypos.values()))
    ax.set_yticklabels([f"{r.name} (prio {r.prio})" for r in order])
    ax.set_title(title, fontsize=10)
    ax.grid(axis="x", alpha=0.35)
    ax.set_ylim(-0.7, len(order) - 0.3)


def resource_legend(fig):
    items = [Patch(color="#2ca02c", label="running (no resource)"),
             Patch(color="#ff7f0e", label="running in critical section"),
             Patch(color="#9467bd", label="critical section at inherited priority"),
             Patch(color="#d62728", alpha=0.35, label="blocked on the resource")]
    fig.legend(handles=items, loc="lower center", ncol=4, fontsize=8)
