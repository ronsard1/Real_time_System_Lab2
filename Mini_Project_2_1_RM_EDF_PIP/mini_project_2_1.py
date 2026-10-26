"""
Mini Project 2.1 - Simulating RM, EDF and Priority Inheritance in Python.

Runs the procedure steps 2-10 and the four lab tasks, prints every result and writes the
figures to figures/ next to this script. Run from the repository root:
    python Mini_Project_2_1_RM_EDF_PIP/mini_project_2_1.py
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

import matplotlib.pyplot as plt  # noqa: E402

from common.rts_sim import (Task, RTask, total_utilization, rm_bound, hyperperiod, rta,  # noqa: E402
                            simulate, plot_two, simulate_resources, draw_resource_timeline,
                            resource_legend, segments)

FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)


def line(c="-"):
    print(c * 72)


# ------------------------------------------------------------------ steps 2-3: utilization tests
def utilization_report(tasks, label):
    U = total_utilization(tasks)
    n = len(tasks)
    b = rm_bound(n)
    print(f"{label}")
    for t in tasks:
        print(f"   {t.name}: C={t.C}, T={t.T}, u={t.U:.4f}")
    print(f"   U = {U:.3f}   rm_bound({n}) = {b:.3f}   "
          f"RM sufficient test: {'PASS' if U <= b else 'INCONCLUSIVE (U > bound)'}   "
          f"EDF U <= 1: {'PASS' if U <= 1 else 'FAIL'}")
    return U, b


line("=")
print("Steps 2-3: utilization and bounds")
line("=")
three = [Task("t1", 1, 4), Task("t2", 2, 6), Task("t3", 2, 12)]
four = three + [Task("t4", 1, 8)]
utilization_report(three, "Section 2.2.2 three-task set")
utilization_report(four, "Section 2.3.2 four-task set (t4 added)")

# ------------------------------------------------------------------ steps 4-6: RM and EDF
H = hyperperiod(four)
rm = simulate(four, H, "rm")
edf = simulate(four, H, "edf")
line()
print(f"Steps 4-6: simulation over one hyperperiod H = {H} ms")
print("  RM  timeline:", "".join((n[-1] if n else ".") for n in rm.timeline))
print("  EDF timeline:", "".join((n[-1] if n else ".") for n in edf.timeline))
print(f"  RM : misses={len(rm.misses)}, context switches={rm.switches}, preemptions={rm.preemptions}")
print(f"  EDF: misses={len(edf.misses)}, context switches={edf.switches}, preemptions={edf.preemptions}")
for t in four:
    print(f"  worst observed response {t.name}: RM {rm.max_response(t.name)}  EDF {edf.max_response(t.name)}")
plot_two(four, rm, edf, f"Rate Monotonic (H = {H} ms)", "Earliest Deadline First",
         os.path.join(FIG, "fig_rm_vs_edf.png"),
         "Four-task set: t1(1,4) t2(2,6) t3(2,12) t4(1,8), U = 0.875")

# ------------------------------------------------------------------ step 7: RTA
line()
print("Step 7: Response Time Analysis (RM priorities)")
rta_rows = rta(four, "rm")
for r in rta_rows:
    print(f"  {r['name']}: iterations {r['trace']} -> R = {r['R']}, D = {r['D']}, "
          f"{'schedulable' if r['ok'] else 'NOT schedulable'}")

# ------------------------------------------------------------------ steps 8-10: inversion and PIP
line()
print("Steps 8-10: priority inversion and the Priority Inheritance Protocol")
L = RTask("L", 1, 0, [("cs", 6, "R")])
M = RTask("M", 2, 2, [("cpu", 4)])
Hh = RTask("H", 3, 1, [("cs", 2, "R")])
scenario = [Hh, M, L]
inv_none, st_none = simulate_resources(scenario, "none", 20)
inv_pip, st_pip = simulate_resources(scenario, "pip", 20)


def inv_summary(st, label):
    print(f"  {label}: H finishes at t={st['H']['finish']}, H blocked {st['H']['blocked_ticks']} ticks, "
          f"M finishes at t={st['M']['finish']}, L finishes at t={st['L']['finish']}")
    return st["H"]["blocked_ticks"]


b_none = inv_summary(st_none, "no protocol")
b_pip = inv_summary(st_pip, "PIP        ")
print(f"  Bound (one critical section of L, Section 2.7.1) = 6 ticks; "
      f"remaining part after H arrives = 5 ticks")

fig, axes = plt.subplots(2, 1, figsize=(10, 5.2), sharex=True)
draw_resource_timeline(axes[0], scenario, inv_none,
                       f"No protocol: H blocked for {b_none} ticks (L's critical section + all of M)")
draw_resource_timeline(axes[1], scenario, inv_pip,
                       f"Priority Inheritance: H blocked for {b_pip} ticks (rest of L's critical section only)")
axes[1].set_xlabel("time (ticks)")
axes[1].set_xticks(range(0, 15))
axes[1].set_xlim(0, 14)
resource_legend(fig)
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(os.path.join(FIG, "fig_inversion_vs_pip.png"), dpi=160)
plt.close(fig)

# ------------------------------------------------------------------ lab task 1: Deadline Monotonic
line("=")
print("Lab task 1: Deadline Monotonic versus Rate Monotonic")
implicit = four
same = simulate(implicit, H, "rm").timeline == simulate(implicit, H, "dm").timeline
print(f"  implicit-deadline set (D = T): DM schedule identical to RM? {same}")
cons = [Task("tA", 1, 5, 5), Task("tB", 2, 8, 2), Task("tC", 1, 10, 10)]
Hc = hyperperiod(cons)
rm_c = simulate(cons, Hc, "rm")
dm_c = simulate(cons, Hc, "dm")
print("  constrained set: tA(C=1,T=5,D=5) tB(C=2,T=8,D=2) tC(C=1,T=10,D=10)")
print(f"  RM priority order: {[t.name for t in sorted(cons, key=lambda t: t.T)]}  misses={len(rm_c.misses)}")
print(f"  DM priority order: {[t.name for t in sorted(cons, key=lambda t: t.D)]}  misses={len(dm_c.misses)}")
for p in ("rm", "dm"):
    rows = rta(cons, p)
    print(f"  RTA under {p.upper()}: " + ", ".join(f"{r['name']} R={r['R']}/D={r['D']}" for r in rows))
plot_two(cons, rm_c, dm_c, "RM priorities (by period) on a constrained-deadline set",
         "DM priorities (by deadline)", os.path.join(FIG, "fig_rm_vs_dm.png"),
         "Constrained deadlines: tA(1,5,5) tB(2,8,2) tC(1,10,10); red v = relative deadline, x = miss")
first_diff = next((i for i, (a, b) in enumerate(zip(rm_c.timeline, dm_c.timeline)) if a != b), None)
print(f"  first tick where the RM and DM schedules differ: t={first_diff}")

# ------------------------------------------------------------------ lab task 2: bound sweep
print("Lab task 2: rm_bound(n) for n = 1..10")
ns = list(range(1, 11))
bounds = [rm_bound(n) for n in ns]
for n, b in zip(ns, bounds):
    print(f"  n={n:2d}  bound={b:.4f}")
print(f"  ln 2 = {math.log(2):.4f}")
fig, ax = plt.subplots(figsize=(7.5, 4))
ax.plot(ns, bounds, "o-", label="RM bound n(2^(1/n) - 1)")
ax.axhline(1.0, color="#2ca02c", label="EDF bound (U <= 1)")
ax.axhline(math.log(2), color="#7f7f7f", ls="--", label="ln 2 = 0.693")
ax.axhline(0.875, color="#d62728", ls=":", label="four-task set, U = 0.875")
for n, b in zip(ns, bounds):
    if n in (1, 2, 3, 4, 10):
        ax.annotate(f"{b:.3f}", (n, b), textcoords="offset points", xytext=(4, 6), fontsize=8)
ax.set_xlabel("number of tasks n")
ax.set_ylabel("utilization bound")
ax.set_xticks(ns)
ax.set_ylim(0.6, 1.05)
ax.grid(alpha=0.35)
ax.legend(fontsize=8, loc="center right")
ax.set_title("Liu & Layland bound versus the EDF bound", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_rm_bound_sweep.png"), dpi=160)
plt.close(fig)

# ------------------------------------------------------------------ lab task 3: LLF thrashing
print("Lab task 3: context switches over one hyperperiod (RM, EDF, LLF)")
# second set with two tasks of equal parameters: their laxities tie repeatedly, which is
# exactly the situation in which LLF thrashes (Section 2.4)
thr = [Task("tX", 3, 8), Task("tY", 3, 8), Task("tZ", 3, 12)]
Ht = hyperperiod(thr)
counts = {}
for set_name, ts, hp in (("four-task set (U=0.875)", four, H), ("tX(3,8) tY(3,8) tZ(3,12) (U=1.0)", thr, Ht)):
    for pol in ("rm", "edf", "llf"):
        r = simulate(ts, hp, pol)
        counts[(set_name, pol.upper())] = (r.switches, r.preemptions, len(r.misses))
        print(f"  {set_name:34s} {pol.upper():4s} dispatches={r.switches:3d} preemptions={r.preemptions:3d} "
              f"misses={len(r.misses)}  {''.join((n[-1] if n else '.') for n in r.timeline)}")
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
for ax, set_name in zip(axes, ("four-task set (U=0.875)", "tX(3,8) tY(3,8) tZ(3,12) (U=1.0)")):
    pols = ["RM", "EDF", "LLF"]
    d = [counts[(set_name, p)][0] for p in pols]
    pr = [counts[(set_name, p)][1] for p in pols]
    xs = range(3)
    ax.bar([x - 0.2 for x in xs], d, 0.4, label="dispatches (job changes)")
    ax.bar([x + 0.2 for x in xs], pr, 0.4, label="preemptions")
    for x in xs:
        ax.text(x - 0.2, d[x] + 0.3, str(d[x]), ha="center", fontsize=9)
        ax.text(x + 0.2, pr[x] + 0.3, str(pr[x]), ha="center", fontsize=9)
        m = counts[(set_name, pols[x])][2]
        if m:
            ax.text(x, max(d[x], pr[x]) + 2.2, f"{m} miss", ha="center", fontsize=8, color="red")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(pols)
    ax.set_title(set_name + ", one hyperperiod", fontsize=10)
    ax.grid(axis="y", alpha=0.35)
axes[0].set_ylabel("count")
axes[0].legend(fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_context_switches.png"), dpi=160)
plt.close(fig)
plot_two(thr, simulate(thr, Ht, "edf"), simulate(thr, Ht, "llf"), "EDF", "Least Laxity First",
         os.path.join(FIG, "fig_edf_vs_llf.png"),
         "Equal-laxity set tX(3,8) tY(3,8) tZ(3,12): LLF alternates every tick")

# ------------------------------------------------------------------ lab task 4: second medium task
print("Lab task 4: a second medium-priority task M2 arrives during M")
M2 = RTask("M2", 2, 4, [("cpu", 3)])
scen2 = [Hh, M, M2, L]
inv2_none, st2_none = simulate_resources(scen2, "none", 25)
inv2_pip, st2_pip = simulate_resources(scen2, "pip", 25)
b2_none = st2_none["H"]["blocked_ticks"]
b2_pip = st2_pip["H"]["blocked_ticks"]
print(f"  no protocol: H blocked {b2_none} ticks (was {b_none}); PIP: H blocked {b2_pip} ticks (was {b_pip})")
fig, axes = plt.subplots(2, 1, figsize=(10, 5.8), sharex=True)
draw_resource_timeline(axes[0], scen2, inv2_none, f"No protocol, with M2: H blocked {b2_none} ticks")
draw_resource_timeline(axes[1], scen2, inv2_pip, f"PIP, with M2: H blocked {b2_pip} ticks")
axes[1].set_xlabel("time (ticks)")
axes[1].set_xticks(range(0, 18))
axes[1].set_xlim(0, 17)
resource_legend(fig)
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(os.path.join(FIG, "fig_inversion_two_medium.png"), dpi=160)
plt.close(fig)

# ------------------------------------------------------------------ step 10 table
line("=")
print("Step 10: blocking of H")
print(f"{'scenario':32s}{'no protocol':>14s}{'PIP':>8s}{'bound':>8s}")
print(f"{'H, M, L (Section 2.6.1)':32s}{b_none:>14d}{b_pip:>8d}{6:>8d}")
print(f"{'H, M, M2, L (lab task 4)':32s}{b2_none:>14d}{b2_pip:>8d}{6:>8d}")
print("Figures written to", FIG)
