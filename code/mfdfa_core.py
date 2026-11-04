#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment 01 — Differential geometry of MFDFA profiles.

Question: does the curvature (log-Hessian) of the MFDFA fluctuation surface
G(u,q)=log2 F_q(s), u=log2 s, carry information that h(q) alone does not?

Battery of 11 stochastic processes x 30 realizations, N=2^16.
Works in log-log coordinates (u=log2 s). First order = h(q); second order =
curvature invariants G_uu (departure from scaling), G_uq (scale-moment
coupling), G_qq.

Decisive test: find the pair of processes whose h(q) curves are closest
(most confusable) and check whether curvature separates them.
"""
import numpy as np
import pandas as pd
from numpy.linalg import pinv
from scipy.signal import lfilter
import json, time, os

RNG_MASTER = np.random.default_rng(20260729)
OUT = os.environ.get("OUTDIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"))
os.makedirs(OUT, exist_ok=True)

# ----------------------------------------------------------------------
# 1. PROCESS SIMULATORS
# ----------------------------------------------------------------------
def fgn_circulant(N, H, rng):
    """Fractional Gaussian noise via Davies-Harte circulant embedding.
    Correlation structure ~ gamma(k)=0.5(|k-1|^2H - 2|k|^2H + |k+1|^2H)."""
    k = np.arange(0, N + 1)
    g = 0.5 * (np.abs(k - 1) ** (2 * H) - 2 * np.abs(k) ** (2 * H)
               + np.abs(k + 1) ** (2 * H))
    c = np.concatenate([g, g[-2:0:-1]])            # length 2N
    lam = np.fft.fft(c).real
    lam = np.clip(lam, 0, None)
    Z = rng.standard_normal(2 * N) + 1j * rng.standard_normal(2 * N)
    y = np.fft.fft(np.sqrt(lam) * Z) / np.sqrt(2 * N)
    return y.real[:N]

def stationary_gaussian(cov_seq, N, rng):
    """Stationary Gaussian vector length N with given covariance sequence
    cov_seq[l] (lag l), via circulant embedding. Real part has Var=cov_seq[0]."""
    M = 2 * N
    L = min(len(cov_seq), N)
    c = np.zeros(M)
    c[:L] = cov_seq[:L]
    c[M - L + 1:] = cov_seq[1:L][::-1]
    lam = np.fft.fft(c).real
    lam = np.clip(lam, 0, None)
    Z = rng.standard_normal(M) + 1j * rng.standard_normal(M)
    y = np.fft.fft(np.sqrt(lam) * Z) / np.sqrt(M)
    return y.real[:N]

def mrw_increments(N, lam2, T, rng):
    """Multifractal Random Walk (Bacry-Muzy-Delour) increments.
    dX_k = eps_k * exp(omega_k), omega Gaussian with logarithmic covariance
    Cov(l)=lam2*ln(T/(l+1)) for l<T, mean=-Var/2."""
    lags = np.arange(N)
    cov = np.where(lags < T, lam2 * np.log(T / (lags + 1.0)), 0.0)
    cov = np.clip(cov, 0, None)
    omega = stationary_gaussian(cov, N, rng)
    omega = omega - 0.5 * cov[0]                   # E[exp(omega)] ~ 1
    eps = rng.standard_normal(N)
    return eps * np.exp(omega)

def binomial_cascade(nlevels, p, rng):
    """Random binomial multiplicative cascade, length 2^nlevels.
    Theoretical tau(q) = -log2(p^q + (1-p)^q)."""
    vals = np.array([1.0])
    for _ in range(nlevels):
        n = vals.size
        swap = rng.random(n) < 0.5
        f_left = np.where(swap, 1 - p, p)
        f_right = np.where(swap, p, 1 - p)
        new = np.empty(2 * n)
        new[0::2] = vals * f_left
        new[1::2] = vals * f_right
        vals = new
    return vals * (2.0 ** nlevels)                 # normalise as density

def make_series(name, N, rng):
    if name == "white_noise":
        return rng.standard_normal(N)
    if name == "random_walk":
        return np.cumsum(rng.standard_normal(N))
    if name == "fGn_H0.75":
        return fgn_circulant(N, 0.75, rng)
    if name == "fGn_H0.50":
        return fgn_circulant(N, 0.50, rng)
    if name == "fGn_H0.25":
        return fgn_circulant(N, 0.25, rng)
    if name == "poisson":
        return rng.poisson(5.0, N).astype(float)
    if name == "AR1_phi0.9":
        return lfilter([1.0], [1.0, -0.9], rng.standard_normal(N))
    if name == "binom_p0.3":
        return binomial_cascade(int(np.log2(N)), 0.30, rng)
    if name == "binom_p0.4":
        return binomial_cascade(int(np.log2(N)), 0.40, rng)
    if name == "MRW_l0.03":
        return mrw_increments(N, 0.03, N // 32, rng)
    if name == "MRW_l0.06":
        return mrw_increments(N, 0.06, N // 32, rng)
    raise ValueError(name)

PROCESSES = ["white_noise", "random_walk", "fGn_H0.75", "fGn_H0.50",
             "fGn_H0.25", "poisson", "AR1_phi0.9", "binom_p0.3",
             "binom_p0.4", "MRW_l0.03", "MRW_l0.06"]

# ----------------------------------------------------------------------
# 2. MFDFA  ->  fluctuation surface F_q(s)
# ----------------------------------------------------------------------
def mfdfa_surface(x, scales, qs, m=2):
    N = len(x)
    Y = np.cumsum(x - x.mean())
    Fq = np.full((len(scales), len(qs)), np.nan)
    for si, s in enumerate(scales):
        ns = N // s
        if ns < 4:
            continue
        segs_f = Y[:ns * s].reshape(ns, s)
        segs_b = Y[N - ns * s:].reshape(ns, s)
        segs = np.vstack([segs_f, segs_b])                 # (2ns, s)
        t = np.arange(s)
        V = np.vander(t, m + 1)                             # (s, m+1)
        coeffs = segs @ pinv(V).T
        resid = segs - coeffs @ V.T
        var = np.mean(resid ** 2, axis=1)                  # F^2(nu,s)
        var = np.clip(var, 1e-300, None)
        for qi, q in enumerate(qs):
            if abs(q) < 1e-6:
                Fq[si, qi] = np.exp(0.5 * np.mean(np.log(var)))
            else:
                Fq[si, qi] = (np.mean(var ** (q / 2.0))) ** (1.0 / q)
    return Fq

# ----------------------------------------------------------------------
# 3. SURFACE GEOMETRY  ->  h(q) + curvature invariants
# ----------------------------------------------------------------------
def geometry_features(Fq, scales, qs):
    """Return dict of first-order (h) and second-order (curvature) features."""
    u = np.log2(scales)
    G = np.log2(Fq)                                        # (n_s, n_q)
    good = np.all(np.isfinite(G), axis=1)
    u = u[good]; G = G[good, :]
    nq = len(qs)

    # ---- first order: h(q) = slope of G vs u ----
    hq = np.array([np.polyfit(u, G[:, j], 1)[0] for j in range(nq)])

    # ---- per-q curvature along scale: G_uu(q) via quadratic fit ----
    Guu_q = np.array([2.0 * np.polyfit(u, G[:, j], 2)[0] for j in range(nq)])

    # ---- running exponent: h fitted on short vs long half of scale range ----
    half = len(u) // 2
    hq_short = np.array([np.polyfit(u[:half], G[:half, j], 1)[0] for j in range(nq)])
    hq_long = np.array([np.polyfit(u[half:], G[half:, j], 1)[0] for j in range(nq)])

    # ---- global 2D quadratic surface fit -> average log-Hessian ----
    U, Q = np.meshgrid(u, np.asarray(qs), indexing="ij")
    A = np.column_stack([np.ones(U.size), U.ravel(), Q.ravel(),
                         U.ravel() ** 2, Q.ravel() ** 2, (U * Q).ravel()])
    beta, *_ = np.linalg.lstsq(A, G.ravel(), rcond=None)
    Guu, Gqq, Guq = 2 * beta[3], 2 * beta[4], beta[5]      # avg log-Hessian

    def val_at(arr, qval):
        return float(np.interp(qval, qs, arr))

    qmin, qmax = qs[0], qs[-1]
    feats = {
        # classical
        "h2": val_at(hq, 2.0),
        "h_width": val_at(hq, qmin) - val_at(hq, qmax),
        # curvature invariants (the new information)
        "Guu": Guu, "Gqq": Gqq, "Guq": Guq,
        "Guu_q2": val_at(Guu_q, 2.0),
        "Guu_absmean": float(np.mean(np.abs(Guu_q))),
        # running exponent (crossover signature)
        "dh_scale_q2": val_at(hq_short, 2.0) - val_at(hq_long, 2.0),
        "dh_scale_absmean": float(np.mean(np.abs(hq_short - hq_long))),
    }
    return feats, hq, Guu_q

# ----------------------------------------------------------------------
# 3b. SCALE-RANGE SAFETY FOR NEGATIVE MOMENTS
# ----------------------------------------------------------------------
def longest_constant_run(x):
    """Length of the longest run of identical consecutive values."""
    x = np.asarray(x)
    idx = np.flatnonzero(np.concatenate([[True], np.diff(x) != 0, [True]]))
    return int(np.max(np.diff(idx)))

def safe_smin(x, smin):
    """Minimum scale above the longest constant run (power of 2), so that no window has
    zero variance and negative moments stay finite (Kluszczynski et al. 2025)."""
    run = longest_constant_run(x)
    return int(max(smin, 2 ** int(np.ceil(np.log2(run + 1)))))

# ----------------------------------------------------------------------
# 4. RUN THE FULL BATTERY
# ----------------------------------------------------------------------
def main():
    N = 2 ** 16
    n_real = 30
    m = 2
    qs = np.arange(-5.0, 5.0001, 0.5)
    scales = np.unique(np.floor(
        2 ** np.linspace(np.log2(16), np.log2(N // 4), 26)).astype(int))

    print(f"N={N}  realizations={n_real}  scales={len(scales)}  q={len(qs)}")
    t0 = time.time()
    rows, hq_store, guu_store = [], {}, {}
    for name in PROCESSES:
        rng = np.random.default_rng(abs(hash(name)) % (2 ** 32))
        hq_list, guu_list = [], []
        for r in range(n_real):
            x = make_series(name, N, rng)
            Fq = mfdfa_surface(x, scales, qs, m=m)
            feats, hq, guu_q = geometry_features(Fq, scales, qs)
            feats["process"] = name
            feats["realization"] = r
            rows.append(feats)
            hq_list.append(hq); guu_list.append(guu_q)
        hq_store[name] = np.array(hq_list)
        guu_store[name] = np.array(guu_list)
        print(f"  {name:14s} done  ({time.time()-t0:6.1f}s)  "
              f"h2={np.mean([r['h2'] for r in rows if r['process']==name]):.3f}  "
              f"width={np.mean([r['h_width'] for r in rows if r['process']==name]):.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "features.csv"), index=False)
    np.savez(os.path.join(OUT, "curves.npz"),
             qs=qs, scales=scales,
             **{f"hq__{k}": v for k, v in hq_store.items()},
             **{f"guu__{k}": v for k, v in guu_store.items()})
    print(f"Saved features.csv and curves.npz  ({time.time()-t0:.1f}s total)")

if __name__ == "__main__":
    main()
