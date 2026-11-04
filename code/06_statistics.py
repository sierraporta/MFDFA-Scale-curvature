#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Summary statistics quoted in Sec. 5.1 and Sec. 5.6: bootstrap CIs for AUCs and leave-one-out
balanced accuracies, dispersion of the recovered exponents, effect size of AR(1) vs fGn(0.75) in G_uu,
|G_uu| separation of AR(1) from genuine cascades, MRW integral-scale sweep, shuffle retention.
Writes results/uncertainty.json."""
import os, json, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
import importlib.util
spec = importlib.util.spec_from_file_location("mfdfa_core", os.path.join(HERE, "mfdfa_core.py"))
E = importlib.util.module_from_spec(spec); spec.loader.exec_module(E)
rng = np.random.default_rng(0)

def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg); allv = np.concatenate([pos, neg])
    order = np.argsort(allv); r = np.empty(len(allv)); r[order] = np.arange(1, len(allv)+1)
    a = (r[:len(pos)].sum() - len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg))
    return max(a, 1-a)

def auc_ci(pos, neg, B=3000):
    pos, neg = np.asarray(pos), np.asarray(neg)
    bs = [auc(rng.choice(pos, len(pos)), rng.choice(neg, len(neg))) for _ in range(B)]
    return round(auc(pos, neg), 3), (round(np.percentile(bs, 2.5), 3), round(np.percentile(bs, 97.5), 3))

def loo_outcomes(X, y):
    X = (X - X.mean(0)) / (X.std(0) + 1e-12); cls = np.unique(y); corr = np.zeros(len(X), bool)
    for k in range(len(X)):
        m = np.ones(len(X), bool); m[k] = False
        cent = {c: X[m & (y == c)].mean(0) for c in cls}
        corr[k] = min(cent, key=lambda c: np.sum((X[k]-cent[c])**2)) == y[k]
    return corr, y

def bal_acc(corr, y):
    return float(np.mean([corr[y == c].mean() for c in np.unique(y)]))

def bal_acc_ci(corr, y, B=3000):
    cls = np.unique(y); pt = bal_acc(corr, y); bs = []
    idx = {c: np.where(y == c)[0] for c in cls}
    for _ in range(B):
        recs = [corr[rng.choice(idx[c], len(idx[c]))].mean() for c in cls]
        bs.append(np.mean(recs))
    return round(pt, 3), (round(np.percentile(bs, 2.5), 3), round(np.percentile(bs, 97.5), 3))

out = {}

# ---- Exp 01: exponent recovery dispersion + LOO CIs ----
df = pd.read_csv(os.path.join(RES, "features.csv"))
dat = np.load(os.path.join(RES, "curves.npz")); qs = dat["qs"]
procs = list(dict.fromkeys(df["process"]))
out["h2_recovery"] = {p: [round(df[df.process==p].h2.mean(), 3), round(df[df.process==p].h2.std(), 3)]
                      for p in ["white_noise","random_walk","fGn_H0.75","binom_p0.3","MRW_l0.06"]}
y = df["process"].values
hq_full = np.vstack([dat[f"hq__{p}"][r] for p in procs for r in range(dat[f"hq__{p}"].shape[0])])
y_hq = np.array([p for p in procs for _ in range(dat[f"hq__{procs[0]}"].shape[0])])
curv = np.vstack([df[df.process==p][["Guu","Guq","dh_scale_q2"]].values for p in procs])
sets = {"h2_only": (df[["h2"]].values, y),
        "h2_plus_curv": (df[["h2","Guu","Guq","dh_scale_q2"]].values, y),
        "hq_full": (hq_full, y_hq),
        "hq_full_plus_curv": (np.hstack([hq_full, curv]), y_hq)}
out["exp01_loo_balacc"] = {k: dict(zip(["point","ci95"], bal_acc_ci(*loo_outcomes(X, yy))))
                           for k, (X, yy) in sets.items()}

# ---- Exp 02: separatrix AUC CI (AR vs genuine, on -Guu) ----
sw = pd.read_csv(os.path.join(RES, "sweep_features.csv"))
ar = -sw[sw.family=="AR"]["Guu"].values
gen = -sw[sw.kind=="genuine"]["Guu"].values
pt, ci = auc_ci(ar, gen); out["exp02_separatrix_auc"] = {"point": pt, "ci95": ci,
    "n_AR": int(len(ar)), "n_genuine": int(len(gen))}


# ---- AR(1) vs fGn(0.75): effect size in G_uu (Cohen's d, pooled sd) ----
a_, b_ = df[df.process=="AR1_phi0.9"].Guu.values, df[df.process=="fGn_H0.75"].Guu.values
sp_ = np.sqrt(((len(a_)-1)*a_.var(ddof=1) + (len(b_)-1)*b_.var(ddof=1)) / (len(a_)+len(b_)-2))
out["ar_vs_fgn_Guu_cohens_d"] = round(float((a_.mean()-b_.mean())/sp_), 2)
out["ar_vs_fgn_h2"] = [round(float(df[df.process==p].h2.mean()), 3) for p in ["AR1_phi0.9", "fGn_H0.75"]]

# ---- |G_uu|: AR(1) vs genuine cascades; detection at |G_uu| = 0.02 ----
absar, absgen = np.abs(sw[sw.family=="AR"]["Guu"].values), np.abs(sw[sw.kind=="genuine"]["Guu"].values)
abst = np.abs(sw[sw.family=="studentt"]["Guu"].values)
out["absGuu_ar_vs_genuine"] = {"auc": auc(absar, absgen),
    "frac_above_0.02": {"AR": round(float(np.mean(absar > 0.02)), 3), "genuine": round(float(np.mean(absgen > 0.02)), 3),
                        "studentt": round(float(np.mean(abst > 0.02)), 3)},
    "frac_AR_negative": round(float(np.mean(sw[sw.family=="AR"]["Guu"].values < 0)), 3)}

# ---- MRW integral-scale sweep ----
mt = pd.read_csv(os.path.join(RES, "mrw_Tsweep.csv"))
out["mrw_Tsweep"] = {"mean_absGuu_by_T": {int(t): round(float(g.Guu.abs().mean()), 4) for t, g in mt.groupby("T")},
                     "n_above_0.02": int((mt.Guu.abs() > 0.02).sum()), "n": int(len(mt))}

# ---- Exp 02: shuffle-retention mean +/- sd (re-simulated) ----
def retained_stats(name, nsh=20, N=2**16):
    r = np.random.default_rng(42)
    x = E.make_series(name, N, r) if name != "studentt_df3" else r.standard_t(3, N)
    sc = np.unique(np.floor(2**np.linspace(np.log2(16), np.log2(N//4), 26)).astype(int))
    qs2 = np.arange(-5.0, 5.0001, 0.5)
    w0 = E.geometry_features(E.mfdfa_surface(x, sc, qs2), sc, qs2)[0]["h_width"]
    fr = []
    for _ in range(nsh):
        xs = x.copy(); r.shuffle(xs)
        fr.append(E.geometry_features(E.mfdfa_surface(xs, sc, qs2), sc, qs2)[0]["h_width"] / w0)
    return [round(float(np.mean(fr)), 3), round(float(np.std(fr)), 3)]
out["shuffle_retained"] = {n: retained_stats(n) for n in
                           ["studentt_df3","AR1_phi0.9","binom_p0.3","MRW_l0.06"]}

# ---- Exp 04: ECG NSR/APB AUCs + LOO CI ----
ecg = pd.read_csv(os.path.join(RES, "ecg_full.csv"))
A, Bp = ecg[ecg.cls=="NSR"], ecg[ecg.cls=="APB"]
out["ecg_auc"] = {f: dict(zip(["point","ci95"], auc_ci(A[f].values, Bp[f].values)))
                  for f in ["h2","h_width","Guu"]}
yb = ecg["cls"].values
out["ecg_loo_balacc"] = {
    "width_only": dict(zip(["point","ci95"], bal_acc_ci(*loo_outcomes(ecg[["h_width"]].values, yb)))),
    "all": dict(zip(["point","ci95"], bal_acc_ci(*loo_outcomes(
        ecg[["h2","h_width","Guu","Guq","dh_scale_absmean","retained"]].values, yb)))),
}

json.dump(out, open(os.path.join(RES, "uncertainty.json"), "w"), indent=2)
print(json.dumps(out, indent=2))
