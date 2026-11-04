#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""finite-size behaviour of width, |G_uu| and q-asymmetry A.

N in {2^12..2^16}; monofractals (white noise, fGn 0.75) give the spurious width and
the noise floors; MRW is the genuine reference; AR(1) phi=0.9 gives detection power.
Usage: python 12_finite_size.py <log2N>
"""
import os, sys, importlib.util, numpy as np, pandas as pd
from scipy.signal import lfilter
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "results")
spec = importlib.util.spec_from_file_location("mfdfa_core", os.path.join(HERE, "mfdfa_core.py"))
E = importlib.util.module_from_spec(spec); spec.loader.exec_module(E)
QS = np.arange(-5.0, 5.0001, 0.5); NREP = 30

def main(k):
    N = 2 ** k
    sc = np.unique(np.floor(2 ** np.linspace(np.log2(16), np.log2(N // 4), 26)).astype(int))
    gens = {"white": lambda r: r.standard_normal(N),
            "fGn0.75": lambda r: E.fgn_circulant(N, 0.75, r),
            "MRW0.06": lambda r: E.mrw_increments(N, 0.06, N // 32, r),
            "AR0.9": lambda r: lfilter([1.0], [1.0, -0.9], r.standard_normal(N))}
    rows = []
    for name, g in gens.items():
        rng = np.random.default_rng(k * 100 + len(name))
        for rep in range(NREP):
            f, hq, gq = E.geometry_features(E.mfdfa_surface(g(rng), sc, QS), sc, QS)
            rows.append(dict(N=N, proc=name, width=f["h_width"], absGuu=abs(f["Guu"]),
                             A=float(gq[QS < 0].mean() - gq[QS > 0].mean()), nscales=len(sc)))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, f"fsize_{k}.csv"), index=False)

if __name__ == "__main__":
    main(int(sys.argv[1]))
