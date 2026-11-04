#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Manuscript figures (written to figures/), in the order they appear in the paper.

fig1_theory.png           closed-form crossover curvature vs simulation (data from 03_theory.py)
fig2_width_curvature.png  AR(1) vs fGn(0.75) h(q); multifractal width vs signed G_uu for all simulated families
fig3_power.png            detection power vs exponent gap and crossover scale; closed-form prediction
fig4_qprofile.png         G_uu(q) on the full q grid; small-dispersion prediction
fig5_probes.png           Gaussianized width, |G_uu| and q-asymmetry A by mechanism
fig6_chow_flags.png       naive Chow false positives; flag rates mechanism x probe
fig7_real_data.png        real data against length-matched nulls
fig8_ecg.png              ECG NSR vs APB: per-segment distributions with AUC
figS1_finite_size.png     noise floors of |G_uu| and A vs N (not in the manuscript)
"""
import os, glob, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
FIG = os.environ.get("FIGDIR", os.path.join(ROOT, "figures")); os.makedirs(FIG, exist_ok=True)
rd = lambda f: pd.read_csv(os.path.join(RES, f))
cat = lambda pat: pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(RES, pat)))])

# ---------------- common style (matches the styled figures of the manuscript) ----------------
plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.8,
                     "axes.titlesize": 10, "axes.labelsize": 11, "legend.fontsize": 9,
                     "xtick.labelsize": 10, "ytick.labelsize": 10, "savefig.dpi": 300})
W2 = (12, 5.2)                                          # two-panel figure size

# one colour and marker per mechanism, used in every figure
STY = {
    "white":      ("white noise",        "0.55",        "o"),
    "fGn0.75":    ("fGn(0.75)",          "0.35",        "s"),
    "rw":         ("random walk",        "0.75",        "D"),
    "poisson":    ("Poisson",            "0.65",        "v"),
    "MRW0.13":    ("MRW",                "tab:green",   "o"),
    "binom0.30":  ("binomial p=0.30",    "darkgreen",   "s"),
    "binom0.42":  ("binomial p=0.42",    "yellowgreen", "s"),
    "studentt3":  ("Student-t",          "tab:purple",  "^"),
    "AR0.9":      ("AR(1)",              "tab:red",     "o"),
    "AR_fGn0.75": ("AR(1)+fGn",          "darkred",     "s"),
    "SUPERPOS3":  ("steepening c=3",     "tab:blue",    "o"),
    "SUPERPOS6":  ("steepening c=6",     "tab:blue",    "D"),
    "SUPERPOS12": ("steepening c=12",    "lightskyblue","s"),
    "PERIODIC256":("periodic",           "tab:brown",   "P"),
    "VARSWITCH8": ("variance switching", "magenta",     "X"),
    "CUSPS":      ("isolated cusps",     "tab:orange",  "v"),
    "SPIKES":     ("isolated spikes",    "goldenrod",   "^"),
}
lab = lambda k: STY[k][0]; col = lambda k: STY[k][1]; mk = lambda k: STY[k][2]
NULLS = ["white", "fGn0.75", "MRW0.13", "binom0.30", "binom0.42", "studentt3"]
ARTS = ["AR0.9", "AR_fGn0.75", "PERIODIC256", "SUPERPOS3", "SUPERPOS12", "VARSWITCH8", "CUSPS", "SPIKES"]

def save(fig, name):
    fig.savefig(os.path.join(FIG, name), bbox_inches="tight"); plt.close(fig); print("wrote", name)

def auc(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return (np.sum(a[:, None] > b[None, :]) + .5 * np.sum(a[:, None] == b[None, :])) / (a.size * b.size)


# ---------- 1. closed-form crossover curvature ----------
th = np.load(os.path.join(RES, "theory_curves.npz"))
fig, ax = plt.subplots(1, 2, figsize=W2)
ax[0].plot(th["log2s"], th["slope"], "o-", ms=3, color="tab:blue", label=r"local slope $G_u=h_{\mathrm{local}}(u)$")
for h, t, ha, dy in [(float(th["H1"]), "$h_1=0.9$", "left", -0.025), (float(th["H2"]), "$h_2=0.5$", "right", 0.01)]:
    ax[0].axhline(h, ls=":", color="gray")
    ax[0].text(th["log2s"][0] if ha == "left" else th["log2s"][-1], h + dy, t, fontsize=9, ha=ha)
ax[0].set_xlabel(r"$\log_2 s$"); ax[0].set_ylabel(r"running exponent $h_{\mathrm{local}}$")
ax[0].set_title(r"Local slope of $\mathrm{fGn}(0.9)+12\,\mathrm{fGn}(0.5)$"); ax[0].legend()
ax[1].plot(th["log2s"], th["guu_closed"], "-", lw=2.5, color="tab:red", label=r"closed form $2(\delta h)^2w_1w_2$")
ax[1].plot(th["log2s"], th["guu_emp"], "o", ms=4, color="k", label=r"measured $G_{uu}$, superposition")
ax[1].plot(th["log2s"], th["guu_single"], "s", ms=4, color="0.55", label=r"measured $G_{uu}$, single fGn")
ax[1].axhline(0, color="k", lw=.6)
ax[1].set_xlabel(r"$\log_2 s$"); ax[1].set_ylabel(r"$G_{uu}$ (natural-log units)")
ax[1].set_title(r"Scale curvature: closed form and simulation"); ax[1].legend()
fig.tight_layout(); save(fig, "fig1_theory.png")

# ---------- 2. crux: matched h(2), distinct curvature; all families ----------
cv = np.load(os.path.join(RES, "curves.npz"), allow_pickle=True); qs = cv["qs"]
bat = rd("features.csv"); fam = cat("families_*.csv"); gcs = cat("gc_*.csv")
fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.2), gridspec_kw={"width_ratios": [1, 1.15]})
for key, k in [("AR1_phi0.9", "AR0.9"), ("fGn_H0.75", "fGn0.75")]:
    H = cv["hq__" + key]; m, s = H.mean(0), H.std(0)
    ax[0].plot(qs, m, color=col(k), lw=2, label=lab(k)); ax[0].fill_between(qs, m - s, m + s, color=col(k), alpha=.2)
ax[0].axvline(2, color="k", ls=":", lw=1); ax[0].set_xlabel("$q$"); ax[0].set_ylabel("$h(q)$")
ax[0].set_title("$h(q)$ of AR(1) and fGn(0.75): equal $h(2)$"); ax[0].legend()
groups = [  # (key, dataframe of (width, Guu))
    ("white", bat[bat.process == "white_noise"]), ("rw", bat[bat.process == "random_walk"]),
    ("fGn0.75", bat[bat.process == "fGn_H0.75"]), ("poisson", bat[bat.process == "poisson"]),
    ("MRW0.13", fam[fam.family == "MRW"]), ("binom0.30", fam[fam.family == "binom"]),
    ("studentt3", fam[fam.family == "studentt"]), ("AR0.9", fam[fam.family == "AR"]),
    ("AR_fGn0.75", fam[fam.family == "AR_fGn"]), ("SUPERPOS6", fam[fam.family == "SUPERPOS"]),
    ("PERIODIC256", fam[fam.family == "PERIODIC"]), ("VARSWITCH8", fam[fam.family == "VARSWITCH"]),
    ("CUSPS", gcs[gcs.family == "CUSPS"]), ("SPIKES", gcs[gcs.family == "SPIKES"])]
gl = {"MRW0.13": "MRW (all $\\lambda^2$)", "binom0.30": "binomial (all $p$)", "studentt3": "Student-t (all $\\nu$)",
      "AR0.9": "AR(1) (all $\\phi$)", "AR_fGn0.75": "AR(1)+fGn", "SUPERPOS6": "steepening (all $c$)",
      "PERIODIC256": "periodic (all periods)"}
for k, d in groups:
    w = d["h_width"] if "h_width" in d else d["width"]
    ax[1].scatter(w, d["Guu"], s=14, alpha=.55, color=col(k), marker=mk(k), label=gl.get(k, lab(k)), edgecolors="none")
thr = 0.021
ax[1].axhspan(-thr, thr, color="0.85", alpha=.6, zorder=0)
ax[1].set_yscale("symlog", linthresh=0.01); ax[1].axhline(0, color="k", lw=.6)
ax[1].set_xlabel(r"multifractal width $\Delta h$"); ax[1].set_ylabel(r"$G_{uu}$ (base-2 units)")
ax[1].set_title(r"Width and signed curvature for all simulated families ($N=2^{16}$)")
ax[1].legend(fontsize=8, ncol=1, loc="upper left", bbox_to_anchor=(1.01, 1.0), markerscale=1.4, frameon=False)
ax[1].text(0.98, 0.02, "shaded: $|G_{uu}|<0.021$ (5% false positives)", transform=ax[1].transAxes,
           ha="right", va="bottom", fontsize=8)
fig.tight_layout(); save(fig, "fig2_width_curvature.png")

# ---------- 1. detection power ----------
g, r = rd("power_gap.csv"), rd("power_ref.csv")
thr = np.percentile(r[r.family != "studentt"].absG, 95)
det = g.groupby(["gap", "u_star"]).absG.apply(lambda s: np.mean(s > thr)).unstack()
fig, ax = plt.subplots(1, 2, figsize=W2)
im = ax[0].imshow(det.values, origin="lower", cmap="viridis", vmin=0, vmax=1, aspect="auto")
ax[0].set_xticks(range(det.shape[1])); ax[0].set_xticklabels([f"$2^{{{c}}}$" for c in det.columns])
ax[0].set_yticks(range(det.shape[0])); ax[0].set_yticklabels([f"{v:.2f}" for v in det.index])
for i in range(det.shape[0]):
    for j in range(det.shape[1]):
        ax[0].text(j, i, f"{det.values[i, j]:.2f}", ha="center", va="center", fontsize=8,
                   color="w" if det.values[i, j] < .6 else "k")
ax[0].grid(False); ax[0].set_xlabel(r"crossover scale $s^*$ (band $2^4$ to $2^{14}$)")
ax[0].set_ylabel(r"exponent gap $\delta h$"); ax[0].set_title("Fraction detected at 5% false positives")
fig.colorbar(im, ax=ax[0], fraction=.046)
U = np.log2(np.unique(np.floor(2 ** np.linspace(4, 14, 26)).astype(int)))
def closed(gap, us, h2=0.5):              # band-averaged curvature of the ideal two-power-law form
    G = 0.5 * np.log2(2 ** (2 * (h2 + gap) * (U - us)) + 2 ** (2 * h2 * (U - us)))
    return abs(2 * np.polyfit(U, G, 2)[0])
m = g.groupby(["gap", "u_star"]).absG.agg(["mean", "std"])
xx = np.linspace(0.05, 0.45, 41)
for us, c in zip([7, 9, 11], ["tab:blue", "tab:green", "tab:red"]):
    d = m.xs(us, level="u_star")
    ax[1].errorbar(d.index, d["mean"], yerr=d["std"], fmt="o", color=c, capsize=2, ms=5, label=f"simulation, $s^*=2^{{{us}}}$")
    ax[1].plot(xx, [closed(v, us) for v in xx], "--", color=c, lw=1.2, label=f"closed form, $s^*=2^{{{us}}}$")
ax[1].axhline(thr, color="k", ls=":", lw=1, label="5% threshold")
ax[1].set_xlabel(r"exponent gap $\delta h$"); ax[1].set_ylabel(r"$|G_{uu}|$ (base-2 units)")
ax[1].set_title(r"Band-averaged $|G_{uu}|$: simulation (mean $\pm$ sd) and closed form")
ax[1].legend(fontsize=8, ncol=2)
fig.tight_layout(); save(fig, "fig3_power.png")

# ---------- 2. three probes by mechanism ----------
gc = gcs[gcs.family.isin(NULLS + ARTS)]; fams = NULLS + ARTS
N0 = gc[gc.family.isin(NULLS)]; tG, tA = np.percentile(N0.absGuu, 95), np.percentile(N0.A, 95)
floor_w = np.percentile(gc[gc.family.isin(["white", "fGn0.75"])].width_gauss, 95)
q = lambda s, p: np.percentile(s, p)
fig, ax = plt.subplots(1, 2, figsize=(12, 5.6))
for f in fams:
    d = gc[gc.family == f]
    for a, xc in [(ax[0], "width_gauss"), (ax[1], "A")]:
        x0, y0 = d[xc].median(), d.absGuu.median()
        a.errorbar(x0, y0, xerr=[[x0 - q(d[xc], 25)], [q(d[xc], 75) - x0]],
                   yerr=[[y0 - q(d.absGuu, 25)], [q(d.absGuu, 75) - y0]], fmt=mk(f), color=col(f), ms=8,
                   mec="k", mew=.5, elinewidth=1, capsize=0, alpha=.9)
for a in ax:
    a.axhline(tG, ls="--", c="k", lw=.8); a.set_yscale("log"); a.set_ylim(1e-3, 0.5)
    a.set_ylabel(r"$|G_{uu}|$ (median, interquartile range)")
ax[0].axvline(floor_w, ls=":", c="k", lw=.8); ax[0].set_xlim(-0.03, 0.32)
ax[0].set_xlabel(r"Gaussianized width $\Delta h_{\mathrm{G}}$")
ax[0].set_title(r"Gaussianized width and $|G_{uu}|$")
ax[1].axvline(tA, ls="--", c="k", lw=.8); ax[1].set_xlim(-0.04, 0.11)
ax[1].set_xlabel("$q$-asymmetry $A$"); ax[1].set_title(r"$q$-asymmetry and $|G_{uu}|$")
handles = [Line2D([], [], ls="", marker=mk(f), color=col(f), mec="k", mew=.5, ms=8, label=lab(f)) for f in fams]
fig.legend(handles=handles, loc="lower center", ncol=7, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.09, 1, 1)); save(fig, "fig5_probes.png")

# ---------- 3. Chow and flag matrix ----------
fp = [np.mean(gc[gc.family == f].chow_p < .05) for f in NULLS]
tC = np.percentile(N0.chowF, 95)
rows = ARTS + ["studentt3", "MRW0.13", "binom0.30", "binom0.42"]
cols = [r"width not temporal" "\n" r"($\Delta h_{\mathrm{G}}\leq$ floor)", r"$|G_{uu}|$", "Chow\n(calibrated)", "$A$"]
Mx = np.array([[np.mean(gc[gc.family == f].width_gauss <= floor_w), np.mean(gc[gc.family == f].absGuu > tG),
                np.mean(gc[gc.family == f].chowF > tC), np.mean(gc[gc.family == f].A > tA)] for f in rows])
fig, ax = plt.subplots(1, 2, figsize=(12, 5.4), gridspec_kw={"width_ratios": [1, 1.25]})
ax[0].bar(range(len(NULLS)), fp, color=[col(f) for f in NULLS], edgecolor="k", lw=.5)
ax[0].axhline(.05, c="k", ls="--", lw=1, label="nominal 5%")
ax[0].set_xticks(range(len(NULLS))); ax[0].set_xticklabels([lab(f) for f in NULLS], rotation=35, ha="right")
ax[0].set_ylim(0, 1.05); ax[0].set_ylabel("rejection rate at nominal $p<0.05$")
ax[0].set_title("Chow test with the $F(2,n-4)$ reference, scaling processes"); ax[0].legend()
nowidth = [gc[gc.family == f].width.median() <= floor_w for f in rows]   # no spurious width to attribute
Mm = np.ma.masked_array(Mx, mask=np.zeros_like(Mx, bool)); Mm.mask[:, 0] = nowidth
cm = plt.get_cmap("Blues").copy(); cm.set_bad("0.88")
im = ax[1].imshow(Mm, cmap=cm, vmin=0, vmax=1, aspect="auto")
for i in range(Mx.shape[0]):
    for j in range(Mx.shape[1]):
        t = "no width" if (j == 0 and nowidth[i]) else f"{Mx[i, j]:.2f}"
        ax[1].text(j, i, t, ha="center", va="center", fontsize=8.5, color="w" if (Mx[i, j] > .6 and t != "no width") else "k")
ax[1].set_xticks(range(len(cols))); ax[1].set_xticklabels(cols, fontsize=9)
ax[1].set_yticks(range(len(rows))); ax[1].set_yticklabels([lab(f) for f in rows])
ax[1].axhline(len(ARTS) - .5, color="k", lw=1.5); ax[1].grid(False)
ax[1].set_title("Fraction of realizations flagged by each probe")
fig.colorbar(im, ax=ax[1], fraction=.046)
fig.tight_layout(); save(fig, "fig6_chow_flags.png")

# ---------- 4. q profiles on the full grid, with the small-dispersion prediction ----------
QG = np.arange(-5, 5.0001, 0.5)
qt = {f: rd(f"qtheory_{f}.csv") for f in ["AR0.9", "SUPERPOS6", "PERIODIC256", "VARSWITCH8", "MRW0.13",
                                              "binom0.30", "studentt3"]}
fig, ax = plt.subplots(1, 2, figsize=W2)
for f, d in qt.items():
    M = np.array([d[f"Guu_q{v:+.1f}"] for v in QG]).T
    ax[0].plot(QG, M.mean(0), marker=mk(f), ms=5, color=col(f), label=lab(f))
    ax[0].fill_between(QG, M.mean(0) - M.std(0), M.mean(0) + M.std(0), color=col(f), alpha=.15)
ax[0].axhline(0, c="k", lw=.6); ax[0].set_xlabel("$q$"); ax[0].set_ylabel(r"$G_{uu}(q)$ (base-2 units)")
ax[0].set_title(r"Measured $G_{uu}(q)$ (mean $\pm$ sd, 20 realizations)"); ax[0].legend(fontsize=8, ncol=2)
for f in ["AR0.9", "SUPERPOS6", "PERIODIC256", "VARSWITCH8"]:
    d = qt[f]; M = np.array([d[f"Guu_q{v:+.1f}"] for v in QG]).T; P = np.array([d[f"pred_q{v:+.1f}"] for v in QG]).T
    ax[1].plot(QG, M.mean(0), marker=mk(f), ms=5, color=col(f), ls="-", label=f"{lab(f)}, measured")
    ax[1].plot(QG, P.mean(0), color=col(f), ls="--", lw=1.5, label=f"{lab(f)}, prediction")
ax[1].axhline(0, c="k", lw=.6); ax[1].set_xlabel("$q$"); ax[1].set_ylabel(r"$G_{uu}(q)$ (base-2 units)")
ax[1].set_title("Small-dispersion prediction from the window-variance dispersion")
ax[1].legend(fontsize=7.5, ncol=2)
fig.tight_layout(); save(fig, "fig4_qprofile.png")

# ---------- 5. finite size ----------
fs = cat("fsize_*.csv"); rows5 = []
for Nn, d in fs.groupby("N"):
    ref = d[d.proc != "AR0.9"]
    rows5.append((Nn, np.percentile(ref.absGuu, 95), np.percentile(ref.A, 95),
                  d[d.proc == "white"].width.mean(), d[d.proc == "AR0.9"].absGuu.mean()))
fsz = np.array(rows5)
fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.loglog(fsz[:, 0], fsz[:, 1], "o-", color="k", label=r"95% floor of $|G_{uu}|$")
ax.loglog(fsz[:, 0], fsz[:, 2], "s-", color="magenta", label="95% floor of $A$")
ax.loglog(fsz[:, 0], fsz[:, 3], "^-", color="0.55", label="spurious width, white noise")
ax.loglog(fsz[:, 0], fsz[:, 4], "d--", color="tab:red", label=r"AR(1) $|G_{uu}|$")
ax.set_xlabel("series length $N$"); ax.legend(fontsize=8); ax.set_title("Finite-size noise floors")
fig.tight_layout(); save(fig, "figS1_finite_size.png")

# ---------- 6. real data ----------
sn = rd("scalingnull_real.csv").set_index("series"); ecg = rd("calib_ecg.csv")
order = [("sun_monthly", "sunspots\nmonthly"), ("sun_daily", "sunspots\ndaily"),
         ("sp_monthly", "S&P 500\nmonthly"), ("sp_daily", "S&P 500\ndaily")]
ratio = [sn.loc[k, "absGuu"] / sn.loc[k, "scal_null_95"] for k, _ in order]
fig, ax = plt.subplots(1, 2, figsize=W2)
ax[0].bar([l for _, l in order], ratio, color=["tab:orange", "tab:orange", "tab:cyan", "tab:cyan"], edgecolor="k", lw=.5)
ax[0].axhline(1, c="k", ls="--", lw=1)
for i, (k, _) in enumerate(order):
    ax[0].text(i, ratio[i] + 0.08, f"p = {sn.loc[k, 'p_scal']:.3f}", ha="center", va="bottom", fontsize=9)
ax[0].set_ylim(0, max(ratio) * 1.15)
ax[0].set_ylabel(r"$|G_{uu}|$ / 95th percentile of the scaling null")
ax[0].set_title("Crossover test against length-matched scaling nulls")
xe = np.arange(2); e = ecg.set_index("series").loc[["ecg_NSR", "ecg_APB"]]
ax[1].bar(xe - .2, e.frac_Guu, .4, color="tab:red", edgecolor="k", lw=.5, label=r"$|G_{uu}|$ above the 95% null")
ax[1].bar(xe + .2, e.frac_gauss_above_floor, .4, color="tab:green", edgecolor="k", lw=.5,
          label=r"$\Delta h_{\mathrm{G}}$ above the Gaussian floor")
for i in range(2):
    ax[1].text(i - .2, e.frac_Guu.iloc[i] + .01, f"{e.frac_Guu.iloc[i]:.2f}", ha="center", fontsize=9)
    ax[1].text(i + .2, e.frac_gauss_above_floor.iloc[i] + .01, f"{e.frac_gauss_above_floor.iloc[i]:.2f}", ha="center", fontsize=9)
ax[1].set_xticks(xe); ax[1].set_xticklabels(["ECG NSR (n=283)", "ECG APB (n=66)"])
ax[1].set_ylim(0, 1.3); ax[1].set_ylabel("fraction of segments"); ax[1].legend(loc="upper center", ncol=2, fontsize=8.5)
ax[1].set_title("ECG segments: crossover flag and Gaussianized width")
fig.tight_layout(); save(fig, "fig7_real_data.png")

# ---------- 7. ECG NSR vs APB ----------
gr = rd("gc_real.csv"); nsr, apb = gr[gr.series == "ecg_NSR"], gr[gr.series == "ecg_APB"]
feats = [("h2", "$h(2)$"), ("A", "$A$"), ("absGuu", r"$|G_{uu}|$"), ("width", r"$\Delta h$"),
         ("width_gauss", r"$\Delta h_{\mathrm{G}}$")]
fig, ax = plt.subplots(1, len(feats), figsize=(13, 3.8))
for a, (c, lb) in zip(ax, feats):
    bp = a.boxplot([nsr[c], apb[c]], tick_labels=["NSR", "APB"], widths=.6, showfliers=False, patch_artist=True,
                   medianprops=dict(color="k"))
    for p, cc in zip(bp["boxes"], ["tab:blue", "tab:orange"]): p.set_facecolor(cc); p.set_alpha(.55)
    v = auc(nsr[c], apb[c]); a.set_title(f"{lb}   AUC = {max(v, 1 - v):.2f}")
fig.tight_layout(); save(fig, "fig8_ecg.png")
