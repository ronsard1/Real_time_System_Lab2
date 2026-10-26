"""
Mini Project 2.3 - Partitioned multiprocessor scheduling simulator.

Runs FF, FFD, BFD and WF on five hand-built task sets with m = 4 cores, prints and saves a
comparison table, runs the per-core RM simulator on one successful partition, and adds a
randomized acceptance-ratio experiment (UUniFast task sets) to back up the comparison.

    python Mini_Project_2_3_Partitioning/mini_project_2_3.py
"""
import csv
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))

import matplotlib.pyplot as plt  # noqa: E402

from partition import HEURISTICS, min_cores  # noqa: E402
from common.rts_sim import Task, rm_bound, hyperperiod, simulate, segments, task_colors  # noqa: E402

FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
M = 4


def tset(prefix, pairs):
    return [Task(f"{prefix}{i + 1}", c, t) for i, (c, t) in enumerate(pairs)]


SETS = {
    "A balanced (16 small tasks)": tset("a", [(3, 20), (2, 16), (4, 25), (3, 20), (7, 50), (3, 20), (2, 16), (4, 25),
                                            (3, 20), (7, 50), (3, 20), (2, 16), (4, 25), (3, 20), (7, 50), (3, 20)]),
    "B skewed (3 large + 10 small)": tset("b", [(1, 20), (2, 25), (6, 10), (1, 10), (2, 25), (11, 20), (1, 20),
                                              (2, 25), (13, 25), (1, 10), (1, 20), (2, 25), (1, 20)]),
    "C tight (U near the core ceiling)": tset("c", [(50, 100), (45, 100), (55, 100), (40, 100), (50, 100), (45, 100),
                                                    (42, 100), (48, 100)]),
    "D adversarial: 5 tasks just over 0.5": tset("d", [(13, 25)] * 5),
    "E adversarial order: small first": tset("e", [(1, 20)] * 8 + [(6, 10)] * 4),
}

rows = []
results = {}
print(f"m = {M} cores; admission: load + u <= rm_bound(n + 1) on the core")
for sname, tasks in SETS.items():
    U = sum(t.U for t in tasks)
    umax = max(t.U for t in tasks)
    print("=" * 78)
    print(f"{sname}: n = {len(tasks)}, U = {U:.3f} ({100 * U / M:.1f}% of m), max u_i = {umax:.3f}, "
          f"lower bound ceil(U) = {math.ceil(U - 1e-9)} cores")
    for hname, h in HEURISTICS.items():
        r = h(tasks, M)
        results[(sname, hname)] = r
        loads = [c.load for c in r.cores]
        ns = [len(c.tasks) for c in r.cores]
        mc = min_cores(tasks, h)
        print(f"  {hname:4s} {'OK  ' if r.success else 'FAIL'} cores used {r.cores_used}  "
              f"per-core U {['%.3f' % x for x in loads]} (n {ns})  "
              f"placed U {r.placed_utilization:.3f} = {100 * r.placed_utilization / M:.1f}% of m  "
              f"unplaced {[t.name for t in r.unplaced]}  min cores needed {mc}")
        rows.append({"set": sname, "heuristic": hname, "success": r.success, "cores_used": r.cores_used,
                     "per_core_U": " / ".join(f"{x:.3f}" for x in loads),
                     "per_core_bound": " / ".join(f"{rm_bound(n):.3f}" if n else "-" for n in ns),
                     "placed_U_pct_of_m": round(100 * r.placed_utilization / M, 1),
                     "total_U_pct_of_m": round(100 * U / M, 1),
                     "unplaced": " ".join(t.name for t in r.unplaced),
                     "min_cores": mc, "lower_bound": math.ceil(U - 1e-9)})

with open(os.path.join(HERE, "results_2_3.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

# ---------------------------------------------------------------- per-core utilization chart
fig, axes = plt.subplots(1, len(SETS), figsize=(15, 3.6), sharey=True)
for ax, sname in zip(axes, SETS):
    for k, hname in enumerate(HEURISTICS):
        r = results[(sname, hname)]
        for c_i, c in enumerate(r.cores):
            x = k * (M + 1) + c_i
            ax.bar(x, c.load, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"][k], edgecolor="black", linewidth=0.4)
            if c.tasks:
                ax.plot([x - 0.4, x + 0.4], [rm_bound(len(c.tasks))] * 2, color="black", lw=1.2)
        if not r.success:
            ax.text(k * (M + 1) + 1.5, 0.93, "FAIL", ha="center", color="red", fontsize=8, fontweight="bold")
    ax.set_xticks([k * (M + 1) + 1.5 for k in range(4)])
    ax.set_xticklabels(list(HEURISTICS))
    ax.set_title(sname.split(" (")[0].split(":")[0] + ("" if ":" not in sname else ":" + sname.split(":")[1]),
                 fontsize=8.5)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.35)
axes[0].set_ylabel("core utilization (4 bars = cores 1-4)")
fig.suptitle("Per-core utilization by heuristic (black tick = RM bound for that core's task count)", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_per_core_utilization.png"), dpi=160)
plt.close(fig)

# ---------------------------------------------------------------- per-core RM simulation (Gantt)
sname = "B skewed (3 large + 10 small)"
part = results[(sname, "FFD")]
H = hyperperiod(SETS[sname])
print("=" * 78)
print(f"Per-core RM simulation of set B under FFD over H = {H} ms")
fig, axes = plt.subplots(M, 1, figsize=(12, 1.25 * M + 1.2), sharex=True)
colors = task_colors([t.name for t in SETS[sname]])
all_ok = True
for i, (ax, core) in enumerate(zip(axes, part.cores)):
    if not core.tasks:
        ax.set_yticks([])
        ax.set_ylabel(f"core {i + 1}\n(empty)", rotation=0, ha="right", va="center", fontsize=8)
        continue
    res = simulate(core.tasks, H, "rm")
    ok = not res.misses
    all_ok &= ok
    segs = segments(res.timeline)
    for name, sg in segs.items():
        if name is None:
            ax.broken_barh(sg, (0.1, 0.8), facecolors="#eeeeee", hatch="///", edgecolor="#bbbbbb", linewidth=0.3)
        else:
            ax.broken_barh(sg, (0.1, 0.8), facecolors=colors[name], edgecolor="black", linewidth=0.3)
            for s, ln in sg:
                if ln >= 3:
                    ax.text(s + ln / 2, 0.5, name, ha="center", va="center", fontsize=6.5, color="white")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=colors[t.name], label=t.name) for t in core.tasks],
              loc="center left", bbox_to_anchor=(1.005, 0.5), fontsize=7, frameon=False)
    ax.set_yticks([])
    lab = ", ".join(f"{t.name}({t.C}/{t.T})" for t in core.tasks)
    ax.set_ylabel(f"core {i + 1}", rotation=0, ha="right", va="center", fontsize=9)
    ax.set_title(f"core {i + 1}: U = {core.load:.3f} <= bound {rm_bound(len(core.tasks)):.3f}; {lab}; "
                 f"{'no misses' if ok else str(len(res.misses)) + ' misses'}", fontsize=8, loc="left")
    print(f"  core {i + 1}: {lab}  U = {core.load:.3f}  misses = {len(res.misses)}  "
          f"max response = {{{', '.join(f'{t.name}: {res.max_response(t.name)}' for t in core.tasks)}}}")
axes[-1].set_xlabel("time (ms)")
axes[-1].set_xlim(0, H)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_per_core_gantt_setB_FFD.png"), dpi=160)
plt.close(fig)
print(f"  every core meets every deadline: {all_ok}")


# ---------------------------------------------------------------- randomized acceptance ratio
def uunifast(n, U, rng):
    us, s = [], U
    for i in range(1, n):
        nxt = s * rng.random() ** (1.0 / (n - i))
        us.append(s - nxt)
        s = nxt
    us.append(s)
    return us


rng = random.Random(2026)
N_TASKS, SETS_PER_POINT = 12, 400
Us = [round(1.0 + 0.2 * k, 2) for k in range(13)]           # 1.0 .. 3.4 on 4 cores
acc = {h: [] for h in HEURISTICS}
for U in Us:
    ok = {h: 0 for h in HEURISTICS}
    made = 0
    while made < SETS_PER_POINT:
        us = uunifast(N_TASKS, U, rng)
        if max(us) > 1.0:
            continue
        made += 1
        tasks = [Task(f"r{i}", 1, 1) for i in range(N_TASKS)]
        for t, u in zip(tasks, us):
            t.C, t.T = round(u * 10000), 10000          # fine-grained utilizations
            if t.C == 0:
                t.C = 1
        for h, fn in HEURISTICS.items():
            ok[h] += fn(tasks, M).success
    for h in HEURISTICS:
        acc[h].append(ok[h] / SETS_PER_POINT)
print("=" * 78)
print(f"Acceptance ratio, {N_TASKS} UUniFast tasks, {SETS_PER_POINT} sets per point, m = {M}")
print("  U     " + "".join(f"{h:>8s}" for h in HEURISTICS))
for i, U in enumerate(Us):
    print(f"  {U:<5.1f} " + "".join(f"{acc[h][i]:>8.3f}" for h in HEURISTICS))
with open(os.path.join(HERE, "results_2_3_acceptance.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["U", "U_pct_of_m"] + list(HEURISTICS))
    for i, U in enumerate(Us):
        w.writerow([U, round(100 * U / M, 1)] + [round(acc[h][i], 3) for h in HEURISTICS])
fig, ax = plt.subplots(figsize=(7.5, 4))
for h, mk in zip(HEURISTICS, "osD^"):
    ax.plot(Us, acc[h], marker=mk, ms=4, label=h)
ax.axvline(M * math.log(2), color="#7f7f7f", ls=":", label=f"m ln2 = {M * math.log(2):.2f}")
ax.set_xlabel(f"total utilization U (m = {M} cores)")
ax.set_ylabel("fraction of task sets partitioned")
ax.set_title(f"Acceptance ratio, {N_TASKS} random tasks, {SETS_PER_POINT} sets per point", fontsize=10)
ax.grid(alpha=0.35)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_acceptance_ratio.png"), dpi=160)
plt.close(fig)
print("results written to results_2_3.csv, results_2_3_acceptance.csv; figures to", FIG)
