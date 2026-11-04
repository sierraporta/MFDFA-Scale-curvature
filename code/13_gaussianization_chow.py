#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gaussianization (Kluszczynski et al. 2025), Chow-type crossover test
(Choi 2026) and isolated singularities (Oswiecimka et al. 2020), against |G_uu| and A.

Per realization: width, |G_uu|, q-asymmetry A, Gaussianized-width ratio g = width(gauss)/width,
shuffle retention, and the Chow two-piece statistic at q=2 (breakpoint search, >=5 scales/side)
with its naive F(2, n-4) p-value.
Usage: python 13_gaussianization_chow.py <family>|real
"""
import os, sys, glob, importlib.util, numpy as np, pandas as pd
from scipy.signal import lfilter
from scipy import stats
import scipy.io as sio
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "results"); RAW = os.path.join(ROOT, "RealDatas")
spec = importlib.util.spec_from_file_location("mfdfa_core", os.path.join(HERE, "mfdfa_core.py"))
E = importlib.util.module_from_spec(spec); spec.loader.exec_module(E)

N = 2 ** 16; NREP = 15; NSH = 4

def scales(n, smin=16, k=26):
    return np.unique(np.floor(2 ** np.linspace(np.log2(smin), np.log2(n // 4), k)).astype(int))

def gaussianize(x, unsigned, rng):
    """Rank-map x onto a Gaussian (half-normal if unsigned), keeping the temporal order;
    ties broken at random (Kluszczynski et al. 2025)."""
    n = len(x); order = np.lexsort((rng.random(n), x))
    p = (np.arange(1, n + 1) - 0.5) / n
    target = stats.halfnorm.ppf(p) if unsigned else stats.norm.ppf(p)
    g = np.empty(n); g[order] = target
    return g

def chow(u, g, minside=5):
    """Two-piece linear fit of g(u) vs single line; returns (F, p_naive, u_break)."""
    n = len(u); X1 = np.column_stack([np.ones(n), u])
    rss1 = np.sum((g - X1 @ np.linalg.lstsq(X1, g, rcond=None)[0]) ** 2)
    best, kb = np.inf, None
    for k in range(minside, n - minside + 1):
        r = 0.0
        for sl in (slice(0, k), slice(k, n)):
            Xs = np.column_stack([np.ones(sl.stop - sl.start), u[sl]])
            r += np.sum((g[sl] - Xs @ np.linalg.lstsq(Xs, g[sl], rcond=None)[0]) ** 2)
        if r < best: best, kb = r, k
    F = ((rss1 - best) / 2) / (best / (n - 4))
    return float(F), float(stats.f.sf(F, 2, n - 4)), float(u[kb])

def analyse(x, qs, sc, unsigned, rng, nsh=NSH):
    Fq = E.mfdfa_surface(x, sc, qs)
    f, hq, gq = E.geometry_features(Fq, sc, qs)
    u = np.log2(sc); G2 = np.log2(Fq[:, np.argmin(np.abs(qs - 2))]); ok = np.isfinite(G2)
    Fc, pc, ub = chow(u[ok], G2[ok])
    wg = E.geometry_features(E.mfdfa_surface(gaussianize(x, unsigned, rng), sc, qs), sc, qs)[0]["h_width"]
    sh = []
    for _ in range(nsh):
        y = x.copy(); rng.shuffle(y)
        sh.append(E.geometry_features(E.mfdfa_surface(y, sc, qs), sc, qs)[0]["h_width"])
    w = f["h_width"]
    return dict(width=w, absGuu=abs(f["Guu"]), Guu=f["Guu"], Guq=f["Guq"], h2=f["h2"],
                A=float(gq[qs < 0].mean() - gq[qs > 0].mean()),
                width_gauss=wg, g_ratio=wg / w if w > 0 else np.nan,
                retained=float(np.mean(sh)) / w if w > 0 else np.nan,
                chowF=Fc, chow_p=pc, chow_u=ub)

FAM = {  # name: (generator, unsigned)
    "white": (lambda r: r.standard_normal(N), False),
    "fGn0.75": (lambda r: E.fgn_circulant(N, 0.75, r), False),
    "MRW0.13": (lambda r: E.mrw_increments(N, 0.13, N // 32, r), False),
    "binom0.30": (lambda r: E.binomial_cascade(16, 0.30, r), True),
    "binom0.42": (lambda r: E.binomial_cascade(16, 0.42, r), True),
    "studentt3": (lambda r: r.standard_t(3, N), False),
    "AR0.9": (lambda r: lfilter([1.0], [1.0, -0.9], r.standard_normal(N)), False),
    "AR_fGn0.75": (lambda r: lfilter([1.0], [1.0, -0.9], E.fgn_circulant(N, 0.75, r)), False),
    "SUPERPOS3": (lambda r: E.fgn_circulant(N, 0.9, r) + 3 * E.fgn_circulant(N, 0.5, r), False),
    "SUPERPOS12": (lambda r: E.fgn_circulant(N, 0.9, r) + 12 * E.fgn_circulant(N, 0.5, r), False),
    "PERIODIC256": (lambda r: np.sin(2 * np.pi * np.arange(N) / 256 + r.uniform(0, 6.3)) + r.standard_normal(N), False),
    "VARSWITCH8": (None, False),
    "CUSPS": (None, False),
    "SPIKES": (None, False),
}

def cusps(r, K=5, H=0.2, amp=50.0):
    """Isolated singularities in the profile: |t-tk|^H cusps added to a random walk, then differenced."""
    t = np.arange(N); prof = np.cumsum(r.standard_normal(N))
    for k in r.choice(N, K, replace=False):
        prof += amp * np.abs(t - k) ** H * r.choice([-1, 1])
    return np.diff(prof, prepend=prof[0])

def spikes(r, K=10, amp=30.0):
    """White noise with K isolated large spikes."""
    x = r.standard_normal(N); idx = r.choice(N, K, replace=False)
    x[idx] += amp * r.choice([-1, 1], K); return x

def varswitch(r, ratio=8, L=2000):
    x = np.empty(N); i = 0; hi = r.random() < 0.5
    while i < N:
        n = int(r.exponential(L)) + 1
        x[i:i + n] = r.standard_normal(min(n, N - i)) * (ratio if hi else 1.0); i += n; hi = not hi
    return x

def main(which):
    if which != "real":
        gen, uns = FAM[which]; gen = gen or {'VARSWITCH8': varswitch, 'CUSPS': cusps, 'SPIKES': spikes}[which]
        rng = np.random.default_rng(abs(hash(which)) % 2 ** 32); qs = np.arange(-5, 5.0001, 0.5); sc = scales(N)
        rows = [dict(analyse(gen(rng), qs, sc, uns, rng), family=which, rep=k) for k in range(NREP)]
        pd.DataFrame(rows).to_csv(os.path.join(OUT, f"gc_{which}.csv"), index=False); return
    rng = np.random.default_rng(11); rows = []
    q5, q4 = np.arange(-5, 5.0001, 0.5), np.arange(-4, 4.0001, 0.5)
    d = pd.read_csv(os.path.join(RAW, "Sunspot", "SN_d_tot_V2.0-2.csv"), sep=";", header=None)
    daily = d[d[0] >= 1849][4].values.astype(float)
    m = pd.read_csv(os.path.join(RAW, "Sunspot", "SN_m_tot_V2.0.csv"), sep=";", header=None)[3].values.astype(float)
    rows.append(dict(series="sun_daily", **analyse(daily, q5, scales(len(daily), E.safe_smin(daily, 16)), True, rng, 10)))
    rows.append(dict(series="sun_monthly", **analyse(m, q5, scales(len(m), E.safe_smin(m, 16)), True, rng, 20)))
    raw = pd.ExcelFile(os.path.join(RAW, "StockMarkets", "ie_data.xls")).parse("Data", header=None).iloc[8:]
    P = pd.to_numeric(raw[1], errors="coerce").dropna().values
    Pd = pd.to_numeric(pd.read_csv(os.path.join(RAW, "StockMarkets", "SP500.csv")).SP500, errors="coerce").dropna().values
    for name, Pr in [("sp_monthly", P), ("sp_daily", Pd)]:
        r = np.diff(np.log(Pr)); r = r[np.isfinite(r)]
        rows.append(dict(series=name, **analyse(r, q4, scales(len(r), E.safe_smin(r, 8), 18), False, rng, 20)))
    for cls, folder in [("NSR", "1 NSR"), ("APB", "2 APB")]:
        for f in sorted(glob.glob(os.path.join(RAW, "ECG", folder, "*.mat"))):
            x = sio.loadmat(f)["val"].ravel().astype(float)
            if x.size < 3000: continue
            rows.append(dict(series="ecg_" + cls, **analyse(x, q4, scales(x.size, E.safe_smin(x, 16), 20), False, rng, 2)))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "gc_real.csv"), index=False)

if __name__ == "__main__":
    main(sys.argv[1])
