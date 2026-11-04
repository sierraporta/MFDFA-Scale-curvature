#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reproduce all results and manuscript figures from the raw data.

Usage:  python code/run_all.py         (from the repository root or anywhere)

Runs every experiment in dependency order, writing all intermediate data to
`results/` and every figure to `figures/`. Raw data are read from `RealDatas/`.
"""
import os, sys, subprocess, time
from pathlib import Path

HERE = Path(__file__).resolve().parent          # code/
ROOT = HERE.parent                               # repository root
RESULTS = ROOT / "results"; RESULTS.mkdir(exist_ok=True)
FIGS = ROOT / "figures";  FIGS.mkdir(exist_ok=True)
RAW = ROOT / "RealDatas"

env = os.environ.copy()
env.update(
    PYTHONHASHSEED="0",                     # make hash()-based process seeds deterministic
    OMP_NUM_THREADS="1",
    OUTDIR=str(RESULTS),
    FIGDIR=str(FIGS),
    DATADIR=str(RAW / "Sunspot"),          # SILSO sunspot CSVs
    ECGDIR=str(RAW / "ECG"),               # MIT-BIH ECG .mat folders
    SPDIR=str(RAW / "StockMarkets"),       # Shiller ie_data.xls + FRED SP500.csv
)

# (script, args, label) in dependency order
GC_FAMS = ["white", "fGn0.75", "MRW0.13", "binom0.30", "binom0.42", "studentt3", "AR0.9", "AR_fGn0.75",
           "SUPERPOS3", "SUPERPOS12", "PERIODIC256", "VARSWITCH8", "CUSPS", "SPIKES"]
QT_FAMS = ["white", "fGn0.75", "AR0.9", "PERIODIC256", "VARSWITCH8", "MRW0.13", "binom0.30", "studentt3",
           "SUPERPOS3", "SUPERPOS6", "SUPERPOS12"]
STEPS = [
    ("01_battery.py", [], "eleven-process battery"),
    ("02_sweep.py", [], "parametric sweeps"),
    ("03_theory.py", [], "closed form: symbolic check and simulation"),
    ("04_mrw_integral_scale.py", [], "MRW integral-scale sweep"),
    ("05_ecg_features.py", [], "ECG segment features"),
    ("06_statistics.py", [], "bootstrap confidence intervals and summary statistics"),
    *[("07_crossover_families.py", [g], f"crossover families ({g})") for g in ("genuine", "cross")],
    *[("08_detection_power.py", [g], f"detection power ({g})") for g in ("ref", "gap")],
    *[("09_shuffle_null.py", [g], f"shuffle null, real data ({g})") for g in ("sun", "sp")],
    ("10_scaling_null.py", ["real"], "scaling null, real data"),
    *[("11_nonstationarity.py", [f], f"nonstationarity and mixtures ({f})")
      for f in ("MRW", "binom", "studentt", "VARSWITCH", "MRW_t", "MRW_AR")],
    *[("12_finite_size.py", [str(k)], f"finite size N=2^{k}") for k in range(12, 17)],
    *[("13_gaussianization_chow.py", [f], f"Gaussianization and Chow test ({f})") for f in GC_FAMS + ["real"]],
    ("14_sunspots_min_scale.py", [], "sunspots and the minimum scale"),
    *[("15_real_calibration.py", [g], f"length-matched calibration ({g})") for g in ("sp", "ecg")],
    *[("16_moment_dependence.py", [f], f"G_uu(q) and small-dispersion approximation ({f})") for f in QT_FAMS],
    ("17_manuscript_figures.py", [], "manuscript figures"),
]

def main():
    t0 = time.time()
    start = sys.argv[1] if len(sys.argv) > 1 else None          # optional: resume from a script name
    active = start is None
    for script, args, label in STEPS:
        active = active or script == start
        if not active: continue
        print(f"\n=== {label}  [{script} {' '.join(args)}] ===", flush=True)
        r = subprocess.run([sys.executable, script, *args], cwd=str(HERE), env=env)
        if r.returncode != 0:
            print(f"\n!!! FAILED at {script} {args} (exit {r.returncode})", file=sys.stderr)
            sys.exit(r.returncode)
    print(f"\nAll steps completed in {time.time()-t0:.0f}s.")
    print(f"Data  -> {RESULTS}\nFigures -> {FIGS}")

if __name__ == "__main__":
    main()
