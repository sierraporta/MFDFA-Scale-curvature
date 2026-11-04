#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""broader crossover families + sign-aware and local-peak curvature.

Usage: python 07_crossover_families.py <group>   with group in {cross, genuine}
Writes results/families_<group>.csv
"""
import os, sys, importlib.util, numpy as np, pandas as pd
from scipy.signal import lfilter

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
OUT = os.environ.get("OUTDIR", os.path.join(ROOT, "results"))
spec = importlib.util.spec_from_file_location("mfdfa_core", os.path.join(HERE, "mfdfa_core.py"))
E = importlib.util.module_from_spec(spec); spec.loader.exec_module(E)

N = 2 ** 16
QS = np.arange(-5.0, 5.0001, 0.5)
SCALES = np.unique(np.floor(2 ** np.linspace(np.log2(16), np.log2(N // 4), 26)).astype(int))
NREP = 20

def ar1(x, phi):
    return lfilter([1.0], [1.0, -phi], x)

def gen(family, p, rng):
    if family == "AR":            # flattening crossover (short-range correlation)
        return ar1(rng.standard_normal(N), p)
    if family == "SUPERPOS":      # steepening crossover: fGn(0.9) + c*fGn(0.5)
        return E.fgn_circulant(N, 0.9, rng) + p * E.fgn_circulant(N, 0.5, rng)
    if family == "AR_fGn":        # AR(1) driven by long-memory noise: flattens to H
        return ar1(E.fgn_circulant(N, p, rng), 0.9)
    if family == "PERIODIC":      # sinusoid (period p) + white noise, unit SNR
        t = np.arange(N); ph = rng.uniform(0, 2 * np.pi)
        return np.sin(2 * np.pi * t / p + ph) + rng.standard_normal(N)
    if family == "VARSWITCH":     # nonstationary: variance regimes (ratio p), mean block 2000
        x = np.empty(N); i = 0; hi = rng.random() < 0.5
        while i < N:
            L = int(rng.exponential(2000)) + 1
            x[i:i + L] = rng.standard_normal(min(L, N - i)) * (p if hi else 1.0)
            i += L; hi = not hi
        return x
    if family == "MRW":
        return E.mrw_increments(N, p, N // 32, rng)
    if family == "binom":
        return E.binomial_cascade(int(np.log2(N)), p, rng)
    if family == "studentt":
        return rng.standard_t(p, N)
    raise ValueError(family)

GROUPS = {
    "cross": {"AR": [0.5, 0.7, 0.85, 0.9, 0.95],
              "SUPERPOS": [3, 6, 12, 25],
              "AR_fGn": [0.6, 0.75, 0.9],
              "PERIODIC": [256, 1024, 4096],
              "VARSWITCH": [2, 4, 8]},
    "genuine": {"MRW": [0.02, 0.05, 0.08, 0.13],
                "binom": [0.42, 0.35, 0.30, 0.25],
                "studentt": [3, 4, 6]},
}
KIND = {"AR": "crossover", "SUPERPOS": "crossover", "AR_fGn": "crossover",
        "PERIODIC": "crossover", "VARSWITCH": "nonstationary",
        "MRW": "genuine", "binom": "genuine", "studentt": "fattail"}

def local_curv(u, g, win=7):
    """Sliding quadratic fit -> local d2g/du2 along u."""
    h = win // 2; out = np.full(len(g), np.nan)
    for i in range(h, len(g) - h):
        out[i] = 2 * np.polyfit(u[i - h:i + h + 1], g[i - h:i + h + 1], 2)[0]
    return out

def features(x):
    Fq = E.mfdfa_surface(x, SCALES, QS)
    f, hq, guu_q = E.geometry_features(Fq, SCALES, QS)
    u = np.log2(SCALES); G2 = np.log2(Fq[:, np.argmin(np.abs(QS - 2))])
    ok = np.isfinite(G2); lc = local_curv(u[ok], G2[ok])
    k = np.nanargmax(np.abs(lc))
    f.update(peak_absGuu_q2=float(np.abs(lc[k])), peak_sign=float(np.sign(lc[k])),
             peak_u=float(u[ok][k]))
    for q in (-4, -2, 2, 4):
        f[f"Guu_q{q}"] = float(np.interp(q, QS, guu_q))
    return f

def main(group):
    rows = []
    for fam, params in GROUPS[group].items():
        for p in params:
            rng = np.random.default_rng(abs(hash((fam, p))) % 2 ** 32)
            for r in range(NREP):
                f = features(gen(fam, p, rng))
                f.update(family=fam, param=p, kind=KIND[fam], rep=r)
                rows.append(f)
            print(fam, p, flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, f"families_{group}.csv"), index=False)

if __name__ == "__main__":
    main(sys.argv[1])
