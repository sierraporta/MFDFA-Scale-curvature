#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""scaling (monofractal, correlated) null for |G_uu|.

Null surrogates are exact-scaling Gaussian processes with the same length, scale band
and h(2) as the observed series: fGn(H) if h(2)<1, cumulated fGn(h-1) if h(2)>=1.
Usage: python 10_scaling_null.py <set>   set in {real, ecg_a, ecg_b}
Writes results/scalingnull_<set>.csv
"""
import os, sys, glob, importlib.util, numpy as np, pandas as pd
import scipy.io as sio
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "results"); RAW = os.path.join(ROOT, "RealDatas")
spec = importlib.util.spec_from_file_location("mfdfa_core", os.path.join(HERE, "mfdfa_core.py"))
E = importlib.util.module_from_spec(spec); spec.loader.exec_module(E)

def scales(N, smin, n):
    return np.unique(np.floor(2 ** np.linspace(np.log2(smin), np.log2(N // 4), n)).astype(int))

def feats(x, qs, sc):
    return E.geometry_features(E.mfdfa_surface(x, sc, qs), sc, qs)[0]

def surrogate(N, h, rng):
    if h < 1:
        return E.fgn_circulant(N, float(np.clip(h, 0.05, 0.95)), rng)
    return np.cumsum(E.fgn_circulant(N, float(np.clip(h - 1, 0.05, 0.95)), rng))

def run(name, x, qs, sc, nsur, rng, extra=None):
    f0 = feats(x, qs, sc)
    null = np.array([abs(feats(surrogate(len(x), f0["h2"], rng), qs, sc)["Guu"]) for _ in range(nsur)])
    row = dict(series=name, N=len(x), h2=f0["h2"], absGuu=abs(f0["Guu"]),
               scal_null_med=float(np.median(null)), scal_null_95=float(np.percentile(null, 95)),
               p_scal=float((1 + np.sum(null >= abs(f0["Guu"]))) / (1 + nsur)))
    if extra: row.update(extra)
    return row

def main(which):
    rng = np.random.default_rng(7); rows = []
    if which == "real":
        q5, q4 = np.arange(-5, 5.0001, 0.5), np.arange(-4, 4.0001, 0.5)
        d = pd.read_csv(os.path.join(RAW, "Sunspot", "SN_d_tot_V2.0-2.csv"), sep=";", header=None)
        daily = d[d[0] >= 1849][4].values.astype(float)
        m = pd.read_csv(os.path.join(RAW, "Sunspot", "SN_m_tot_V2.0.csv"), sep=";", header=None)[3].values.astype(float)
        rows.append(run("sun_daily", daily, q5, scales(len(daily), E.safe_smin(daily, 16), 26), 60, rng))
        rows.append(run("sun_monthly", m, q5, scales(len(m), E.safe_smin(m, 16), 26), 200, rng))
        raw = pd.ExcelFile(os.path.join(RAW, "StockMarkets", "ie_data.xls")).parse("Data", header=None).iloc[8:]
        P = pd.to_numeric(raw[1], errors="coerce").dropna().values
        Pd = pd.to_numeric(pd.read_csv(os.path.join(RAW, "StockMarkets", "SP500.csv")).SP500, errors="coerce").dropna().values
        for name, Pr in [("sp_monthly", P), ("sp_daily", Pd)]:
            r = np.diff(np.log(Pr)); r = r[np.isfinite(r)]
            rows.append(run(name, r, q4, scales(len(r), E.safe_smin(r, 8), 18), 300, rng))
    else:
        q4 = np.arange(-4, 4.0001, 0.5); files = []
        for cls, folder in [("NSR", "1 NSR"), ("APB", "2 APB")]:
            files += [(cls, f) for f in sorted(glob.glob(os.path.join(RAW, "ECG", folder, "*.mat")))]
        half = len(files) // 2
        for cls, f in (files[:half] if which == "ecg_a" else files[half:]):
            x = sio.loadmat(f)["val"].ravel().astype(float)
            if x.size < 3000: continue
            rows.append(run(os.path.basename(f), x, q4, scales(x.size, E.safe_smin(x, 16), 20), 30, rng, dict(cls=cls)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, f"scalingnull_{which}.csv"), index=False)
    if which == "real":
        print(df.round(4).to_string())

if __name__ == "__main__":
    main(sys.argv[1])
