#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MRW (lambda^2 = 0.06) with integral scale T from N/128 to N/4: does a genuine multifractal whose
cascade stops inside the band produce scale curvature? -> results/mrw_Tsweep.csv"""
import numpy as np, pandas as pd, os, json, importlib.util

OUT = os.environ.get("OUTDIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"))
spec = importlib.util.spec_from_file_location("exp", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "mfdfa_core.py"))
E = importlib.util.module_from_spec(spec); spec.loader.exec_module(E)

N = 2**16
qs = np.arange(-5.0, 5.0001, 0.5)
scales = np.unique(np.floor(2**np.linspace(np.log2(16), np.log2(N//4), 26)).astype(int))
lam2 = 0.06
Tvals = [N//128, N//64, N//32, N//16, N//8, N//4]   # 512 ... 16384
nrel = 15
rows = []
for T in Tvals:
    rng = np.random.default_rng(1000 + T)
    for r in range(nrel):
        x = E.mrw_increments(N, lam2, T, rng)
        f, _, _ = E.geometry_features(E.mfdfa_surface(x, scales, qs), scales, qs)
        f.update(T=T, realization=r)
        rows.append(f)
    sub = [s for s in rows if s["T"] == T]
    print(f"T={T:6d} (~{T/N:.3f} N)  -Guu={-np.mean([s['Guu'] for s in sub]):+.3f}  "
          f"-Guq={-np.mean([s['Guq'] for s in sub]):+.3f}  width={np.mean([s['h_width'] for s in sub]):.3f}")
df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "mrw_Tsweep.csv"), index=False)

