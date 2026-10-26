"""
sched_tool.py - RM versus EDF schedulability comparison tool (Mini Project 2.2).

Usage (from the repository root):
    python Mini_Project_2_2_Schedulability_Tool/sched_tool.py tasks.csv [--out chart.png] [--horizon N]

CSV columns: name, C, T, D   (D optional; empty or missing means D = T)

Decision layer
  RM / DM : Deadline Monotonic ordering whenever any D < T, otherwise Rate Monotonic.
            Implicit deadlines: Liu & Layland bound first; if it fails, Response Time Analysis.
            Constrained deadlines: the L&L bound assumes D = T, so it is reported for
            reference only and RTA always decides.
  EDF     : implicit deadlines: exact test U <= 1.
            constrained deadlines: U <= 1 is necessary only, so the exact processor-demand
            test (dbf(t) <= t) decides.
  U > 1   : reported as infeasible on one processor for every policy, never a crash.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from common.rts_sim import (Task, total_utilization, rm_bound, hyperperiod, rta,  # noqa: E402
                            edf_demand_test, simulate, plot_two, is_implicit)


def load_csv(path):
    tasks = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            row = {k.strip(): (v.strip() if v else "") for k, v in row.items()}
            D = int(row["D"]) if row.get("D") else None
            tasks.append(Task(row["name"], int(row["C"]), int(row["T"]), D))
    for t in tasks:
        if t.D > t.T:
            raise ValueError(f"{t.name}: D > T (arbitrary deadlines) is outside this tool's model")
        if t.C > t.D:
            print(f"  warning: {t.name} has C > D and can never meet its deadline")
    return tasks


def analyse(tasks):
    n = len(tasks)
    U = total_utilization(tasks)
    implicit = is_implicit(tasks)
    policy = "rm" if implicit else "dm"
    bound = rm_bound(n)
    res = {"n": n, "U": U, "implicit": implicit, "policy": policy, "bound": bound,
           "H": hyperperiod(tasks)}
    res["bound_pass"] = implicit and U <= bound
    if U > 1:
        res["fp_verdict"] = "INFEASIBLE (U > 1)"
        res["rta"] = rta(tasks, policy)
        res["fp_ok"] = False
    elif res["bound_pass"]:
        res["fp_verdict"] = "SCHEDULABLE (utilization bound)"
        res["rta"] = None
        res["fp_ok"] = True
    else:
        rows = rta(tasks, policy)
        res["rta"] = rows
        res["fp_ok"] = all(r["ok"] for r in rows)
        res["fp_verdict"] = ("SCHEDULABLE (exact, RTA)" if res["fp_ok"]
                             else "NOT SCHEDULABLE (RTA: " + ", ".join(r["name"] for r in rows if not r["ok"]) + " misses)")
    if U > 1:
        res["edf_ok"], res["edf_verdict"] = False, "INFEASIBLE (U > 1)"
    elif implicit:
        res["edf_ok"], res["edf_verdict"] = True, "SCHEDULABLE (exact, U <= 1)"
    else:
        ok, t_bad = edf_demand_test(tasks)
        res["edf_ok"] = ok
        res["edf_verdict"] = ("SCHEDULABLE (exact, processor demand)" if ok
                              else f"NOT SCHEDULABLE (dbf({t_bad}) > {t_bad})")
    return res


def report(tasks, res, title=""):
    p = res["policy"].upper()
    out = []
    out.append("=" * 74)
    out.append(f"{title}  ({res['n']} tasks, {'implicit' if res['implicit'] else 'constrained'} deadlines, "
               f"hyperperiod {res['H']})")
    out.append("-" * 74)
    out.append(f"{'task':8s}{'C':>5s}{'T':>6s}{'D':>6s}{'u_i':>9s}")
    for t in tasks:
        out.append(f"{t.name:8s}{t.C:>5d}{t.T:>6d}{t.D:>6d}{t.U:>9.4f}")
    out.append(f"Total U = {res['U']:.4f}   L&L bound n(2^(1/n)-1) = {res['bound']:.4f}   "
               + ("bound test: " + ("PASS" if res["bound_pass"] else "FAIL")
                  if res["implicit"] else "bound not applicable (D < T)"))
    if res["rta"]:
        out.append(f"RTA ({p} priority order):")
        for r in res["rta"]:
            out.append(f"   {r['name']:6s} R = {r['R']:>4d}  D = {r['D']:>4d}  iterations = {r['iterations']:>2d}  "
                       f"{'OK' if r['ok'] else 'MISS'}   trace {r['trace']}")
    out.append(f"{p} verdict : {res['fp_verdict']}")
    out.append(f"EDF verdict: {res['edf_verdict']}")
    text = "\n".join(out)
    print(text)
    return text


def chart(tasks, res, path, horizon=None, title=""):
    H = res["H"]
    hz = horizon or min(H, 500)
    hz = max(hz, max(t.T for t in tasks))
    r1 = simulate(tasks, hz, res["policy"])
    r2 = simulate(tasks, hz, "edf")
    plot_two(tasks, r1, r2, f"{res['policy'].upper()}  (shown {hz} of H = {H})", "EDF", path,
             f"{title}  U = {res['U']:.3f}")
    return r1, r2


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--out", default=None)
    ap.add_argument("--horizon", type=int, default=None)
    a = ap.parse_args()
    tasks = load_csv(a.csv)
    res = analyse(tasks)
    report(tasks, res, os.path.basename(a.csv))
    out = a.out or os.path.splitext(a.csv)[0] + "_chart.png"
    chart(tasks, res, out, a.horizon, os.path.basename(a.csv))
    print("chart written to", out)


if __name__ == "__main__":
    main()
