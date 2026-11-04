#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Experiment 02 — parametric sweep to map the genuine/spurious multifractality
boundary in the curvature plane.

Four families, each swept continuously:
  AR(1) phi        -> spurious multifractality from LINEAR CORRELATION / crossover
  MRW  lambda^2    -> GENUINE multifractality (correlated log-cascade)
  binomial p       -> GENUINE multifractality (multiplicative cascade)
  Student-t df     -> spurious multifractality from FAT TAILS (iid, no correlation)

Question: is there a clean separatrix, and does it distinguish the TWO spurious
mechanisms from the genuine ones, or only the crossover kind?
"""
import numpy as np, pandas as pd, os, time, json
import importlib.util
spec = importlib.util.spec_from_file_location("exp", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "mfdfa_core.py"))
E = importlib.util.module_from_spec(spec); spec.loader.exec_module(E)

OUT = os.environ.get("OUTDIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"))

def make_sweep_series(family, param, N, rng):
    if family == "AR":
        from scipy.signal import lfilter
        return lfilter([1.0], [1.0, -param], rng.standard_normal(N))
    if family == "MRW":
        return E.mrw_increments(N, param, N // 32, rng)
    if family == "binom":
        return E.binomial_cascade(int(np.log2(N)), param, rng)
    if family == "studentt":
        return rng.standard_t(param, N)
    raise ValueError(family)

SWEEP = {
    "AR":       [0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97],
    "MRW":      [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.13],
    "binom":    [0.48, 0.45, 0.42, 0.40, 0.37, 0.35, 0.32, 0.30, 0.27, 0.25],
    "studentt": [30, 12, 8, 6, 5, 4, 3.5, 3],
}
KIND = {"AR": "spurious-correlation", "MRW": "genuine",
        "binom": "genuine", "studentt": "spurious-fattail"}

def main():
    N = 2 ** 16
    n_real = 20
    qs = np.arange(-5.0, 5.0001, 0.5)
    scales = np.unique(np.floor(
        2 ** np.linspace(np.log2(16), np.log2(N // 4), 26)).astype(int))
    t0 = time.time(); rows = []
    for family, params in SWEEP.items():
        for param in params:
            rng = np.random.default_rng(abs(hash((family, param))) % (2 ** 32))
            for r in range(n_real):
                x = make_sweep_series(family, param, N, rng)
                Fq = E.mfdfa_surface(x, scales, qs, m=2)
                feats, hq, guu_q = E.geometry_features(Fq, scales, qs)
                feats.update(family=family, param=float(param),
                             kind=KIND[family], realization=r)
                rows.append(feats)
            sub = [rr for rr in rows if rr["family"] == family and rr["param"] == param]
            print(f"{family:9s} p={param:<5} h2={np.mean([s['h2'] for s in sub]):.3f} "
                  f"width={np.mean([s['h_width'] for s in sub]):.3f} "
                  f"-Guu={-np.mean([s['Guu'] for s in sub]):+.3f} "
                  f"-Guq={-np.mean([s['Guq'] for s in sub]):+.3f} "
                  f"dHsc={np.mean([s['dh_scale_absmean'] for s in sub]):.3f} "
                  f"({time.time()-t0:.0f}s)")
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "sweep_features.csv"), index=False)
    print(f"saved sweep_features.csv ({time.time()-t0:.0f}s)")

if __name__ == "__main__":
    main()
