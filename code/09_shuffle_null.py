#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""surrogate-calibrated curvature test on the real data.

For each series: G_uu (global, same settings as the paper), the shuffle null of
|G_uu| at the same length and scale band, p = P(|G_uu|_null >= |G_uu|), and the
shuffle-retained width fraction. Usage: python 09_shuffle_null.py <set>
set in {sun, sp, ecg_a, ecg_b}  (ECG split in halves for runtime)
"""
import os, sys, glob, importlib.util, numpy as np, pandas as pd
import scipy.io as sio
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "results"); RAW = os.path.join(ROOT, "RealDatas")
spec = importlib.util.spec_from_file_location("mfdfa_core", os.path.join(HERE, "mfdfa_core.py"))
E = importlib.util.module_from_spec(spec); spec.loader.exec_module(E)

def scales(N, smin, n):
    return np.unique(np.floor(2 ** np.linspace(np.log2(smin), np.log2(N // 4), n)).astype(int))

def test(x, qs, sc, nsh, rng):
    f0 = E.geometry_features(E.mfdfa_surface(x, sc, qs), sc, qs)[0]
    nullG, nullW = [], []
    for _ in range(nsh):
        y = x.copy(); rng.shuffle(y)
        f = E.geometry_features(E.mfdfa_surface(y, sc, qs), sc, qs)[0]
        nullG.append(abs(f["Guu"])); nullW.append(f["h_width"])
    nullG = np.array(nullG)
    return dict(N=len(x), Guu=f0["Guu"], absGuu=abs(f0["Guu"]), Guq=f0["Guq"],
                width=f0["h_width"], null_med=float(np.median(nullG)),
                null_95=float(np.percentile(nullG, 95)),
                p=float((1 + np.sum(nullG >= abs(f0["Guu"]))) / (1 + nsh)),
                retained=float(np.mean(nullW) / f0["h_width"]) if f0["h_width"] > 0 else np.nan)

def main(which):
    rng = np.random.default_rng(2026); rows = []
    if which == "sun":
        q = np.arange(-5, 5.0001, 0.5)
        d = pd.read_csv(os.path.join(RAW, "Sunspot", "SN_d_tot_V2.0-2.csv"), sep=";", header=None)
        daily = d[d[0] >= 1849][4].values.astype(float)
        m = pd.read_csv(os.path.join(RAW, "Sunspot", "SN_m_tot_V2.0.csv"), sep=";", header=None)[3].values.astype(float)
        for name, x, nsh in [("sun_daily", daily, 60), ("sun_monthly", m, 200)]:
            rows.append(dict(series=name, **test(x, q, scales(len(x), E.safe_smin(x, 16), 26), nsh, rng)))
    elif which == "sp":
        q = np.arange(-4, 4.0001, 0.5)
        raw = pd.ExcelFile(os.path.join(RAW, "StockMarkets", "ie_data.xls")).parse("Data", header=None).iloc[8:]
        P = pd.to_numeric(raw[1], errors="coerce").dropna().values
        Pd = pd.to_numeric(pd.read_csv(os.path.join(RAW, "StockMarkets", "SP500.csv")).SP500, errors="coerce").dropna().values
        for name, Pr in [("sp_monthly", P), ("sp_daily", Pd)]:
            r = np.diff(np.log(Pr)); r = r[np.isfinite(r)]
            rows.append(dict(series=name, **test(r, q, scales(len(r), E.safe_smin(r, 8), 18), 300, rng)))
    else:
        q = np.arange(-4, 4.0001, 0.5)
        files = []
        for cls, folder in [("NSR", "1 NSR"), ("APB", "2 APB")]:
            files += [(cls, f) for f in sorted(glob.glob(os.path.join(RAW, "ECG", folder, "*.mat")))]
        half = len(files) // 2
        files = files[:half] if which == "ecg_a" else files[half:]
        for cls, f in files:
            x = sio.loadmat(f)["val"].ravel().astype(float)
            if x.size < 3000: continue
            rows.append(dict(series=os.path.basename(f), cls=cls,
                             **test(x, q, scales(x.size, E.safe_smin(x, 16), 20), 40, rng)))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, f"surrogate_{which}.csv"), index=False)
    print(pd.DataFrame(rows).drop(columns=[c for c in ["series"] if which.startswith("ecg")]).describe().T.round(3)
          if which.startswith("ecg") else pd.DataFrame(rows).round(4).to_string())

if __name__ == "__main__":
    main(sys.argv[1])
