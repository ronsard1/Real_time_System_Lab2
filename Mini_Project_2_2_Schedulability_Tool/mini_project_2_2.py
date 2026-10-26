"""
Mini Project 2.2 - runs the schedulability tool on every CSV in tasksets/, checks the RTA
results of the Section 2.3.2 set against hand-computed values, cross-checks every analytical
verdict against a simulation over the hyperperiod, and writes charts plus a summary table.

    python Mini_Project_2_2_Schedulability_Tool/mini_project_2_2.py
"""
import csv
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))

import matplotlib.pyplot as plt  # noqa: E402

from sched_tool import load_csv, analyse, report, chart  # noqa: E402
from common.rts_sim import simulate, rm_bound  # noqa: E402

FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

summary = []
texts = []
for path in sorted(glob.glob(os.path.join(HERE, "tasksets", "*.csv"))):
    name = os.path.splitext(os.path.basename(path))[0]
    tasks = load_csv(path)
    res = analyse(tasks)
    texts.append(report(tasks, res, name))
    hz = min(res["H"], 500)
    r_fp, r_edf = chart(tasks, res, os.path.join(FIG, f"{name}.png"), hz, name)
    # cross-check: analysis must agree with a simulation of the synchronous release (critical
    # instant), which for these task sets is exact over one hyperperiod
    sim_fp_ok = len(r_fp.misses) == 0
    sim_edf_ok = len(r_edf.misses) == 0
    agree = (sim_fp_ok == res["fp_ok"]) and (sim_edf_ok == res["edf_ok"])
    print(f"   simulation over {hz}: {res['policy'].upper()} misses={len(r_fp.misses)}, "
          f"EDF misses={len(r_edf.misses)} -> analysis {'agrees' if agree else 'DISAGREES'}")
    summary.append({
        "set": name, "n": res["n"], "U": round(res["U"], 4), "bound": round(res["bound"], 4),
        "deadlines": "implicit" if res["implicit"] else "constrained",
        "bound_test": ("pass" if res["bound_pass"] else "fail") if res["implicit"] else "n/a",
        "fixed_priority": res["policy"].upper(),
        "fp_verdict": res["fp_verdict"], "edf_verdict": res["edf_verdict"],
        "max_R_over_D": (max(r["R"] / r["D"] for r in res["rta"]) if res["rta"] else None),
        "sim_fp_misses": len(r_fp.misses), "sim_edf_misses": len(r_edf.misses),
        "fp_switches": r_fp.switches, "edf_switches": r_edf.switches, "agree": agree,
    })

# ---------------------------------------------------------------- hand check, Section 2.3.2 set
print("=" * 74)
print("Hand-computed RTA for the Section 2.3.2 set (RM order t1, t2, t4, t3)")
hand = {"t1": 1, "t2": 3, "t4": 4, "t3": 11}
tasks = load_csv(os.path.join(HERE, "tasksets", "set2_section_2_3_2.csv"))
from common.rts_sim import rta  # noqa: E402
for r in rta(tasks, "rm"):
    print(f"   {r['name']}: tool R = {r['R']}, hand R = {hand[r['name']]} -> "
          f"{'match' if r['R'] == hand[r['name']] else 'MISMATCH'}")
    assert r["R"] == hand[r["name"]]

with open(os.path.join(HERE, "results_2_2.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(summary[0]))
    w.writeheader()
    w.writerows(summary)
with open(os.path.join(HERE, "results_2_2.txt"), "w") as f:
    f.write("\n".join(texts) + "\n")

# ---------------------------------------------------------------- overview chart
fig, ax = plt.subplots(figsize=(9, 4))
labels = [s["set"].split("_", 1)[0] for s in summary]
Us = [s["U"] for s in summary]
bs = [s["bound"] for s in summary]
colors = []
for s in summary:
    fp = "SCHEDULABLE" in s["fp_verdict"] and "NOT" not in s["fp_verdict"]
    ed = "SCHEDULABLE" in s["edf_verdict"] and "NOT" not in s["edf_verdict"]
    colors.append("#2ca02c" if fp and ed else ("#ff7f0e" if ed else "#d62728"))
ax.bar(labels, Us, color=colors, edgecolor="black", linewidth=0.5)
ax.scatter(labels, bs, marker="_", s=900, color="black", label="L&L bound for n tasks", zorder=3)
ax.axhline(1.0, color="#7f7f7f", ls="--", label="U = 1")
for i, u in enumerate(Us):
    ax.text(i, u + 0.02, f"{u:.3f}", ha="center", fontsize=8)
from matplotlib.patches import Patch  # noqa: E402
ax.legend(handles=[Patch(color="#2ca02c", label="RM/DM and EDF schedulable"),
                   Patch(color="#ff7f0e", label="EDF only"),
                   Patch(color="#d62728", label="neither"),
                   plt.Line2D([], [], color="black", marker="_", ls="", ms=14, label="L&L bound n(2^(1/n)-1)"),
                   plt.Line2D([], [], color="#7f7f7f", ls="--", label="U = 1")], fontsize=8, loc="upper left")
ax.set_ylabel("total utilization U")
ax.set_ylim(0, 1.4)
ax.set_title("Eight task sets: utilization, L&L bound and verdicts", fontsize=10)
ax.grid(axis="y", alpha=0.35)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "overview_verdicts.png"), dpi=160)
plt.close(fig)
print("results written to results_2_2.csv / results_2_2.txt, charts to", FIG)
