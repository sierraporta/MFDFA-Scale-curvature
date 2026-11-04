#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""length-matched calibration for the short real series.
Scaling surrogates (same N, scale band, h(2)) give null distributions of the Chow F,
|G_uu|, A and of the width of a Gaussian monofractal (floor for the Gaussianized width).
Usage: python 15_real_calibration.py {sp|ecg}
"""
import os, sys, importlib.util, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); OUT = os.path.join(ROOT, "results")
def load(name, path):
    s = importlib.util.spec_from_file_location(name, os.path.join(HERE, path))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
E = load("mfdfa_core", "mfdfa_core.py"); GC = load("gc", "13_gaussianization_chow.py"); SN = load("sn", "10_scaling_null.py")

def null_stats(N, h2, qs, sc, n, rng):
    rows = []
    for _ in range(n):
        y = SN.surrogate(N, h2, rng)
        Fq = E.mfdfa_surface(y, sc, qs); f, _, gq = E.geometry_features(Fq, sc, qs)
        u = np.log2(sc); G2 = np.log2(Fq[:, np.argmin(np.abs(qs - 2))]); ok = np.isfinite(G2)
        rows.append((GC.chow(u[ok], G2[ok])[0], abs(f["Guu"]), gq[qs < 0].mean() - gq[qs > 0].mean(), f["h_width"]))
    return np.array(rows)

def main(which):
    rng = np.random.default_rng(31); real = pd.read_csv(os.path.join(OUT, "gc_real.csv")); out = []
    q4 = np.arange(-4, 4.0001, 0.5)
    if which == "sp":
        lens = {"sp_monthly": 1866, "sp_daily": 2511}
        for s, N in lens.items():
            r = real[real.series == s].iloc[0]; sc = GC.scales(N, 8, 18)
            nl = null_stats(N, r.h2, q4, sc, 300, rng)
            out.append(dict(series=s, N=N, chowF=r.chowF, p_chow=np.mean(nl[:, 0] >= r.chowF),
                            absGuu=r.absGuu, p_Guu=np.mean(nl[:, 1] >= r.absGuu), A=r.A, p_A=np.mean(nl[:, 2] >= r.A),
                            width=r.width, width_gauss=r.width_gauss, floor_width95=np.percentile(nl[:, 3], 95)))
    else:
        e = real[real.series.str.startswith("ecg")]; sc = GC.scales(3600, 16, 20)
        nl = np.vstack([null_stats(3600, h, q4, sc, 60, rng) for h in (0.7, 0.9, 1.1)])
        for c, g in e.groupby("series"):
            out.append(dict(series=c, n=len(g), frac_chow=np.mean(g.chowF > np.percentile(nl[:, 0], 95)),
                            frac_Guu=np.mean(g.absGuu > np.percentile(nl[:, 1], 95)),
                            frac_A_pos=np.mean(g.A > np.percentile(nl[:, 2], 95)),
                            frac_A_neg=np.mean(g.A < np.percentile(nl[:, 2], 5)),
                            width_gauss_med=g.width_gauss.median(), floor_width95=np.percentile(nl[:, 3], 95),
                            frac_gauss_above_floor=np.mean(g.width_gauss > np.percentile(nl[:, 3], 95))))
    df = pd.DataFrame(out); df.to_csv(os.path.join(OUT, f"calib_{which}.csv"), index=False)
    print(df.round(4).T.to_string())

if __name__ == "__main__":
    main(sys.argv[1])
