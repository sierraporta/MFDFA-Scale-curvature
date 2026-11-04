#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""nonstationarity and mixed mechanisms.

Can variance-regime switching (nonstationary) be told apart from genuine cascades,
and how do mixtures (genuine + fat tails, genuine + crossover) classify?
Features: width, global G_uu, G_uq, G_uu(q) profile, q-asymmetry
A = <G_uu(q<0)> - <G_uu(q>0)>, and shuffle-retained width.
Usage: python 11_nonstationarity.py <family>   (one family per call; see CELLS)
"""
import os, sys, importlib.util, numpy as np, pandas as pd
from scipy.signal import lfilter
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "results")
spec = importlib.util.spec_from_file_location("mfdfa_core", os.path.join(HERE, "mfdfa_core.py"))
E = importlib.util.module_from_spec(spec); spec.loader.exec_module(E)

N = 2 ** 16; NREP = 15; NSH = 8
QS = np.arange(-5.0, 5.0001, 0.5)
SC = np.unique(np.floor(2 ** np.linspace(np.log2(16), np.log2(N // 4), 26)).astype(int))

def varswitch(ratio, L, rng):
    x = np.empty(N); i = 0; hi = rng.random() < 0.5
    while i < N:
        n = int(rng.exponential(L)) + 1
        x[i:i + n] = rng.standard_normal(min(n, N - i)) * (ratio if hi else 1.0); i += n; hi = not hi
    return x

def mrw_t(lam2, df, rng):            # MRW with Student-t innovations (genuine + fat tails)
    lags = np.arange(N); T = N // 32
    cov = np.clip(np.where(lags < T, lam2 * np.log(T / (lags + 1.0)), 0.0), 0, None)
    om = E.stationary_gaussian(cov, N, rng) - 0.5 * cov[0]
    return rng.standard_t(df, N) * np.exp(om)

CELLS = {
    "VARSWITCH": [(r, L) for r in (2, 4, 8) for L in (500, 2000, 8000)],
    "MRW": [0.05, 0.13], "binom": [0.30, 0.42], "studentt": [3],
    "MRW_t": [(0.05, 3), (0.13, 3)],                       # genuine + fat tails
    "MRW_AR": [(0.05, 0.9), (0.13, 0.9)],                  # genuine + crossover
}

def gen(fam, p, rng):
    if fam == "VARSWITCH": return varswitch(p[0], p[1], rng)
    if fam == "MRW": return E.mrw_increments(N, p, N // 32, rng)
    if fam == "binom": return E.binomial_cascade(16, p, rng)
    if fam == "studentt": return rng.standard_t(p, N)
    if fam == "MRW_t": return mrw_t(p[0], p[1], rng)
    if fam == "MRW_AR": return lfilter([1.0], [1.0, -p[1]], E.mrw_increments(N, p[0], N // 32, rng))

def feats(x, rng):
    f, hq, gq = E.geometry_features(E.mfdfa_surface(x, SC, QS), SC, QS)
    sh = []
    for _ in range(NSH):
        y = x.copy(); rng.shuffle(y)
        sh.append(E.geometry_features(E.mfdfa_surface(y, SC, QS), SC, QS)[0]["h_width"])
    out = dict(width=f["h_width"], Guu=f["Guu"], Guq=f["Guq"], h2=f["h2"],
               asym=float(gq[QS < 0].mean() - gq[QS > 0].mean()),
               retained=float(np.mean(sh) / f["h_width"]) if f["h_width"] > 0 else np.nan)
    for q in (-4, -2, 2, 4): out[f"Guu_q{q}"] = float(np.interp(q, QS, gq))
    return out

def main(fam):
    rows = []
    for p in CELLS[fam]:
        rng = np.random.default_rng(abs(hash((fam, str(p)))) % 2 ** 32)
        for r in range(NREP):
            rows.append(dict(feats(gen(fam, p, rng), rng), family=fam, param=str(p), rep=r))
        print(fam, p, flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, f"nonstat_{fam}.csv"), index=False)

if __name__ == "__main__":
    main(sys.argv[1])
