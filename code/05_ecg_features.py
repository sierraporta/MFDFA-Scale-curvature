#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Full ECG analysis: all NSR + APB segments.
Per-segment features (h(2), width, curvature, shuffle retention) -> results/ecg_full.csv,
used for the NSR vs APB statistics in uncertainty."""
import numpy as np, pandas as pd, os, glob, json, importlib.util
import scipy.io as sio

OUT = os.environ.get("OUTDIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"))
DATA = os.environ.get("ECGDIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "RealDatas", "ECG"))
spec = importlib.util.spec_from_file_location("exp", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "mfdfa_core.py"))
E = importlib.util.module_from_spec(spec); spec.loader.exec_module(E)

qs = np.arange(-4.0, 4.0001, 0.5)
def scales_for(N, nsc=20):
    return np.unique(np.floor(2**np.linspace(np.log2(16), np.log2(N//4), nsc)).astype(int))
def load(f): return sio.loadmat(f)["val"].ravel().astype(float)
import re
def rec_of(f): return re.match(r"(\d+)m", os.path.basename(f)).group(1)

rng = np.random.default_rng(0)
rows = []
for cls, folder in [("NSR", "1 NSR"), ("APB", "2 APB")]:
    for f in sorted(glob.glob(os.path.join(DATA, folder, "*.mat"))):
        x = load(f)
        if x.size < 3000: continue
        sc = scales_for(x.size)
        feats, hq, _ = E.geometry_features(E.mfdfa_surface(x, sc, qs), sc, qs)
        w0 = feats["h_width"]; xs = x.copy(); rng.shuffle(xs)
        w1, _, _ = E.geometry_features(E.mfdfa_surface(xs, sc, qs), sc, qs)
        feats.update(cls=cls, rec=rec_of(f), retained=(w1["h_width"]/w0 if w0>0 else np.nan))
        rows.append(feats)
df = pd.DataFrame(rows).dropna()
df.to_csv(os.path.join(OUT, "ecg_full.csv"), index=False)
print("segments:", len(df), "| NSR", (df.cls=="NSR").sum(), "APB", (df.cls=="APB").sum())

# ---------------- NSR vs APB discrimination ----------------
def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg); allv = np.concatenate([pos, neg])
    order = np.argsort(allv); ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv)+1)
    a = (ranks[:len(pos)].sum() - len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg))
    return max(a, 1-a)
def cohend(a, b):
    a, b = np.asarray(a), np.asarray(b); s = np.sqrt((a.var(ddof=1)+b.var(ddof=1))/2)
    return (a.mean()-b.mean())/s if s>0 else 0.0
A, B = df[df.cls=="NSR"], df[df.cls=="APB"]
feat_stats = {}
for k in ["h2", "h_width", "Guu", "Guq", "dh_scale_absmean", "retained"]:
    feat_stats[k] = {"AUC": round(auc(A[k], B[k]), 3), "cohen_d": round(cohend(A[k], B[k]), 2)}

def loo_ncc(X, y):
    X = (X - X.mean(0))/(X.std(0)+1e-12); cls = np.unique(y)
    # balanced accuracy
    correct = {c: 0 for c in cls}; tot = {c: 0 for c in cls}
    for k in range(len(X)):
        m = np.ones(len(X), bool); m[k] = False
        cent = {c: X[m & (y==c)].mean(0) for c in cls}
        pred = min(cent, key=lambda c: np.sum((X[k]-cent[c])**2))
        tot[y[k]] += 1; correct[y[k]] += (pred == y[k])
    return np.mean([correct[c]/tot[c] for c in cls])
y = df["cls"].values
sets = {"width_only": df[["h_width"]].values,
        "curvature": df[["Guu", "Guq", "dh_scale_absmean"]].values,
        "h2_plus_curvature": df[["h2", "Guu", "Guq"]].values,
        "all": df[["h2", "h_width", "Guu", "Guq", "dh_scale_absmean", "retained"]].values}
acc = {k: round(loo_ncc(v, y), 3) for k, v in sets.items()}
report = {"n_NSR": int((df.cls=="NSR").sum()), "n_APB": int((df.cls=="APB").sum()),
          "per_feature": feat_stats, "loo_balanced_accuracy": acc}

print(json.dumps(report, indent=2))
json.dump(report, open(os.path.join(OUT, "ecg_full_report.json"), "w"), indent=2)
