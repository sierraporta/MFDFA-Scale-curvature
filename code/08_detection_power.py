#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""detection power vs exponent gap and crossover position.

Steepening crossovers x = fGn(H1) + c*fGn(0.5), gap dh = H1-0.5, crossover placed
at the true scale s* where the two components have equal F^2 (calibrated on unit fGn). Also recomputes genuine references. Stores global |G_uu|
and a regularized local-peak curvature (window 9, top 3 scales excluded).
Usage: python 08_detection_power.py <group> [H1]  group in {gap, ref}; optional H1 runs one gap row
"""
import os, sys, importlib.util, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
OUT = os.environ.get("OUTDIR", os.path.join(ROOT, "results"))
spec = importlib.util.spec_from_file_location("mfdfa_core", os.path.join(HERE, "mfdfa_core.py"))
E = importlib.util.module_from_spec(spec); spec.loader.exec_module(E)

N = 2 ** 16
QS = np.arange(-5.0, 5.0001, 0.5)
SCALES = np.unique(np.floor(2 ** np.linspace(np.log2(16), np.log2(N // 4), 26)).astype(int))
NREP = 20

def local_peak(u, g, win=9, cut=3):
    h = win // 2; uu, gg = u[:len(u) - cut], g[:len(g) - cut]; best = 0.0
    for i in range(h, len(gg) - h):
        c = 2 * np.polyfit(uu[i - h:i + h + 1], gg[i - h:i + h + 1], 2)[0]
        best = max(best, abs(c))
    return best

def feats(x):
    Fq = E.mfdfa_surface(x, SCALES, QS)
    f, _, _ = E.geometry_features(Fq, SCALES, QS)
    G2 = np.log2(Fq[:, np.argmin(np.abs(QS - 2))]); u = np.log2(SCALES)
    return dict(absG=abs(f["Guu"]), Guu=f["Guu"], width=f["h_width"],
                peak=local_peak(u, G2))

def unit_F2(H, rng, n=10):
    """Mean q=2 fluctuation function F^2(s) of unit-variance fGn(H) (DFA-2), n realizations."""
    return np.mean([E.mfdfa_surface(E.fgn_circulant(N, H, rng), SCALES, np.array([2.0]))[:, 0] ** 2
                    for _ in range(n)], axis=0)

def main(group, sub=None):
    rows = []
    if group == "gap":
        H1s = [float(sub)] if sub else [round(0.55 + 0.05 * k, 2) for k in range(9)]
        F2_lo = unit_F2(0.5, np.random.default_rng(7))
        for H1 in H1s:
            F2_hi = unit_F2(H1, np.random.default_rng(int(H1 * 1000)))
            for us in range(6, 13):                    # log2 of the true crossover scale (w1 = w2)
                c = np.sqrt(np.exp(np.interp(us, np.log2(SCALES), np.log(F2_hi / F2_lo))))
                rng = np.random.default_rng(1000 * int(round(H1 * 100)) + us)
                for r in range(NREP):
                    x = E.fgn_circulant(N, H1, rng) + c * E.fgn_circulant(N, 0.5, rng)
                    rows.append(dict(feats(x), family="SUPERPOS", gap=round(H1 - 0.5, 2), u_star=us, c=c, rep=r))
            print("H1", H1, flush=True)
        name = f"power_gap_{sub}.csv" if sub else "power_gap.csv"
        pd.DataFrame(rows).to_csv(os.path.join(OUT, name), index=False); return
    else:
        for fam, p in [("MRW", 0.05), ("MRW", 0.13), ("binom", 0.30), ("binom", 0.25),
                       ("studentt", 3)]:
            rng = np.random.default_rng(abs(hash((fam, p))) % 2 ** 32)
            for r in range(NREP):
                x = (E.mrw_increments(N, p, N // 32, rng) if fam == "MRW" else
                     E.binomial_cascade(16, p, rng) if fam == "binom" else rng.standard_t(p, N))
                rows.append(dict(feats(x), family=fam, param=p, rep=r))
            print(fam, p, flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, f"power_{group}.csv"), index=False)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
