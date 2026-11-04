#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""sunspot analysis with the minimum scale above the longest run of
constant values (daily: 92 zero-days -> smin=128; monthly: 21 -> smin=32), as required
for negative moments (Kluszczynski et al. 2025). Iid and scaling nulls for |G_uu| and A,
Gaussianized width, Chow statistic. Writes results/sun_fixed.csv
"""
import os, importlib.util, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "RealDatas"); OUT = os.path.join(ROOT, "results")
def load(name, path):
    s = importlib.util.spec_from_file_location(name, os.path.join(HERE, path))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
E = load("mfdfa_core", "mfdfa_core.py"); GC = load("gc", "13_gaussianization_chow.py"); SN = load("sn", "10_scaling_null.py")

def longest_run(x):
    idx = np.flatnonzero(np.concatenate([[True], np.diff(x) != 0, [True]])); return int(np.max(np.diff(idx)))

def main():
    rng = np.random.default_rng(2027); qs = np.arange(-5, 5.0001, 0.5); rows = []
    d = pd.read_csv(os.path.join(RAW, "Sunspot", "SN_d_tot_V2.0-2.csv"), sep=";", header=None)
    m = pd.read_csv(os.path.join(RAW, "Sunspot", "SN_m_tot_V2.0.csv"), sep=";", header=None)
    for name, x, nnull in [("sun_daily", d[d[0] >= 1849][4].values.astype(float), 60),
                           ("sun_monthly", m[3].values.astype(float), 200)]:
        smin = E.safe_smin(x, 16)
        sc = np.unique(np.floor(2 ** np.linspace(np.log2(smin), np.log2(len(x) // 4), 26)).astype(int))
        a = GC.analyse(x, qs, sc, True, rng, 10)
        f = lambda y: E.geometry_features(E.mfdfa_surface(y, sc, qs), sc, qs)
        def stats_(y):
            g, _, gq = f(y); return abs(g["Guu"]), gq[qs < 0].mean() - gq[qs > 0].mean()
        iid, scal = [], []
        for _ in range(nnull):
            y = x.copy(); rng.shuffle(y); iid.append(stats_(y))
            scal.append(stats_(SN.surrogate(len(x), a["h2"], rng)))
        iid, scal = np.array(iid), np.array(scal)
        p = lambda null, obs: float((1 + np.sum(null >= obs)) / (1 + len(null)))
        rows.append(dict(series=name, smin=smin, longest_run=longest_run(x), **a,
                         p_Guu_iid=p(iid[:, 0], a["absGuu"]), p_Guu_scal=p(scal[:, 0], a["absGuu"]),
                         scal95_Guu=float(np.percentile(scal[:, 0], 95)),
                         p_A_scal=p(scal[:, 1], a["A"]), scal95_A=float(np.percentile(scal[:, 1], 95))))
    df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, "sun_fixed.csv"), index=False)
    print(df[["series", "smin", "longest_run", "h2", "width", "width_gauss", "absGuu", "Guu", "scal95_Guu", "p_Guu_scal",
              "A", "scal95_A", "p_A_scal", "retained", "chowF"]].round(4).T.to_string())

if __name__ == "__main__":
    main()
