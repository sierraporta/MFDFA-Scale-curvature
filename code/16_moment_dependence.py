#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""moment-order dependence of the scale curvature.

For each family: full profile G_uu(q) on the q grid, and its small-dispersion prediction
    G(u,q) ~ G(u,2) + (q-2)/8 * CV2(u) / ln 2      (base-2 units)
where CV2(u) = Var_nu[F^2(nu,s)] / (E_nu[F^2(nu,s)])^2 is the squared coefficient of
variation of the windowed detrended variances. For the additive superposition the
dispersion constants c1, c2, c12 of the two components and their cross term are also stored.
Usage: python 16_moment_dependence.py <family>
"""
import os, sys, importlib.util, numpy as np, pandas as pd
from numpy.linalg import pinv
from scipy.signal import lfilter
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
OUT = os.environ.get("OUTDIR", os.path.join(ROOT, "results"))
spec = importlib.util.spec_from_file_location("mfdfa_core", os.path.join(HERE, "mfdfa_core.py"))
E = importlib.util.module_from_spec(spec); spec.loader.exec_module(E)

N = 2 ** 16; NREP = 20; M = 2
QS = np.arange(-5.0, 5.0001, 0.5)
SC = np.unique(np.floor(2 ** np.linspace(4, np.log2(N // 4), 26)).astype(int))
U = np.log2(SC)

def resid(x, s):
    Y = np.cumsum(x - x.mean()); ns = N // s
    segs = np.vstack([Y[:ns * s].reshape(ns, s), Y[N - ns * s:].reshape(ns, s)])
    V = np.vander(np.arange(s), M + 1)
    return segs - (segs @ pinv(V).T) @ V.T

def window_var(x):
    return [np.clip(np.mean(resid(x, s) ** 2, axis=1), 1e-300, None) for s in SC]

def surface(vs):
    G = np.empty((len(SC), len(QS)))
    for i, v in enumerate(vs):
        for j, q in enumerate(QS):
            G[i, j] = 0.5 * np.mean(np.log(v)) if abs(q) < 1e-9 else np.log(np.mean(v ** (q / 2))) / q
    return G / np.log(2)

curv = lambda g: 2 * np.polyfit(U, g, 2)[0]

def varswitch(r, ratio=8, L=2000):
    x = np.empty(N); i = 0; hi = r.random() < 0.5
    while i < N:
        n = int(r.exponential(L)) + 1
        x[i:i + n] = r.standard_normal(min(n, N - i)) * (ratio if hi else 1.0); i += n; hi = not hi
    return x

FAM = {
    "white": lambda r: r.standard_normal(N),
    "fGn0.75": lambda r: E.fgn_circulant(N, 0.75, r),
    "AR0.9": lambda r: lfilter([1.0], [1.0, -0.9], r.standard_normal(N)),
    "PERIODIC256": lambda r: np.sin(2 * np.pi * np.arange(N) / 256 + r.uniform(0, 6.3)) + r.standard_normal(N),
    "VARSWITCH8": varswitch,
    "MRW0.13": lambda r: E.mrw_increments(N, 0.13, N // 32, r),
    "binom0.30": lambda r: E.binomial_cascade(16, 0.30, r),
    "studentt3": lambda r: r.standard_t(3, N),
}

def main(fam):
    rng = np.random.default_rng(abs(hash(("qtheory", fam))) % 2 ** 32); rows = []
    for k in range(NREP):
        extra = {}
        if fam.startswith("SUPERPOS"):
            c = float(fam[8:]); x1 = E.fgn_circulant(N, 0.9, rng); x2 = c * E.fgn_circulant(N, 0.5, rng)
            x = x1 + x2; cs = []
            for s in SC:
                r1, r2 = resid(x1, s), resid(x2, s)
                v1, v2, v12 = np.mean(r1 ** 2, 1), np.mean(r2 ** 2, 1), 2 * np.mean(r1 * r2, 1)
                cs.append((np.var(v1) / v1.mean() ** 2, np.var(v2) / v2.mean() ** 2,
                           np.var(v12) / (v1.mean() * v2.mean()), v1.mean() / (v1.mean() + v2.mean())))
            cs = np.array(cs)
            extra = {f"{n}_s{s}": cs[i, j] for i, s in enumerate(SC) for j, n in enumerate(["c1", "c2", "c12", "w1"])}
        else:
            x = FAM[fam](rng)
        vs = window_var(x); G = surface(vs)
        cv2 = np.array([np.var(v) / np.mean(v) ** 2 for v in vs])
        G2 = G[:, np.argmin(np.abs(QS - 2))]
        row = dict(family=fam, rep=k, Guu_global2=curv(G2))
        for j, q in enumerate(QS):
            row[f"Guu_q{q:+.1f}"] = curv(G[:, j])
            row[f"pred_q{q:+.1f}"] = curv(G2 + (q - 2) / 8 * cv2 / np.log(2))
        row.update({f"cv2_s{s}": c for s, c in zip(SC, cv2)}); row.update(extra)
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, f"qtheory_{fam}.csv"), index=False)
    print("done", fam, flush=True)

if __name__ == "__main__":
    main(sys.argv[1])
