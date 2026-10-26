# Real-Time Systems – Lab 2 (Unit 2 Mini Projects)

University of Rwanda, College of Science and Technology, School of ICT, Department of Computer Engineering
Group 3: Ronsard RUGIRANGOGA (223026673), Ivan MULISA (223026695), Dieudonne BYIRINGIRO (223026513)

Pure-Python simulations of the three Unit 2 mini projects. No RTOS, compiler or board is needed.
The written report summarises the results; this repository holds the complete code, every chart
and every printed result, including the ones that are not reproduced in the report.

## Structure

```
common/rts_sim.py                         shared module: Task, rm_bound, RTA, EDF demand test,
                                          tick-based RM/DM/EDF/LLF simulator, PIP simulator, Gantt plots
Mini_Project_2_1_RM_EDF_PIP/
    mini_project_2_1.py                   procedure steps 2-10 and lab tasks 1-4
    results_2_1.txt                       printed output (utilization tests, RTA, blocking table)
    figures/                              RM vs EDF, inversion vs PIP, RM vs DM, bound sweep,
                                          context switches, EDF vs LLF, two medium tasks
Mini_Project_2_2_Schedulability_Tool/
    sched_tool.py                         the tool (CSV in, report + two-panel chart out)
    mini_project_2_2.py                   runs the tool on every task set and checks it
    tasksets/*.csv                        eight task sets (sections 2.2.2 and 2.3.2, U > 1,
                                          constrained deadlines, ten tasks, U = 1, harmonic)
    results_2_2.txt, results_2_2.csv      full reports and summary table
    figures/                              one RM/DM-vs-EDF chart per task set + overview
Mini_Project_2_3_Partitioning/
    partition.py                          FF, FFD, BFD, WF behind one interface
    mini_project_2_3.py                   five test sets on m = 4 cores, per-core Gantt chart,
                                          randomized acceptance-ratio experiment
    results_2_3.txt, results_2_3.csv, results_2_3_acceptance.csv
    figures/
```

## Run

Python 3.10 or later.

```
pip install -r requirements.txt
python Mini_Project_2_1_RM_EDF_PIP/mini_project_2_1.py
python Mini_Project_2_2_Schedulability_Tool/mini_project_2_2.py
python Mini_Project_2_3_Partitioning/mini_project_2_3.py
```

Each script prints its results and rewrites its `figures/` folder. All three finish in a few seconds.
The simulations are deterministic, so the numbers match the report exactly.

The schedulability tool also works on its own with any CSV file (columns `name,C,T,D`, D optional):

```
python Mini_Project_2_2_Schedulability_Tool/sched_tool.py my_tasks.csv --out my_chart.png
```

## Modelling notes

* Time is discrete (1 tick = 1 ms); all tasks are released synchronously at t = 0, which is the
  critical instant for fixed-priority scheduling, so one hyperperiod is enough to confirm the analysis.
* A job that misses its deadline keeps running (it is not dropped), so overload shows as a cascade of misses.
* For constrained deadlines (D < T) the Liu & Layland bound does not apply, so the tool goes straight to
  RTA with Deadline Monotonic priorities, and uses the exact processor-demand test for EDF because
  U <= 1 is only necessary in that case.
* Partitioning admission on each core is `load + u_i <= rm_bound(n + 1)`, recomputed for the core's task
  count, never a flat threshold.
