#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Closed form of the crossover curvature.
(1) SymPy check of G_uu = 2 (h1-h2)^2 w1 w2 for two additive power laws
(2) numerical verification on fGn(0.9) + 12 fGn(0.5) (data for Fig. 1 -> results/theory_curves.npz)."""
import numpy as np, sympy as sp, os, importlib.util

OUT = os.environ.get("OUTDIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"))
spec = importlib.util.spec_from_file_location("exp", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "mfdfa_core.py"))
E = importlib.util.module_from_spec(spec); spec.loader.exec_module(E)

# ----------------------------------------------------------------------
# (1) Symbolic derivation, u = ln s,  F^2 = A1 e^{2 h1 u} + A2 e^{2 h2 u}
# ----------------------------------------------------------------------
u, h1, h2, A1, A2 = sp.symbols("u h1 h2 A1 A2", positive=True)
F2 = A1*sp.exp(2*h1*u) + A2*sp.exp(2*h2*u)
g = sp.Rational(1, 2)*sp.log(F2)           # g = ln F
gp = sp.simplify(sp.diff(g, u))            # local slope  = running exponent
gpp = sp.simplify(sp.diff(g, u, 2))        # curvature G_uu

# express gpp in terms of weights w1,w2
w1 = A1*sp.exp(2*h1*u)/F2
gpp_weight = sp.simplify(gpp - 2*(h1-h2)**2*w1*(1-w1))
print("=== SYMBOLIC ===")
print("G_u  (local slope) =", sp.simplify(gp))
print("G_uu  =", gpp)
print("G_uu - 2(h1-h2)^2 w1 w2  simplifies to:", gpp_weight, " (0 => identity holds)")
print("exact scaling (A2->0): G_uu =", sp.limit(gpp, A2, 0))

# ----------------------------------------------------------------------
# (2) Numerical verification on independent superposition of two fGn
#     x = fGn(H1) + c*fGn(H2)  ->  DFA F2^2(s) ~ A1 s^{2H1} + A2 s^{2H2}
# ----------------------------------------------------------------------
H1, H2, c = 0.9, 0.5, 12.0    # c>>1 places the crossover mid-band (s* ~ c^2.5 ~ 600)
N = 2**16
scales = np.unique(np.floor(2**np.linspace(np.log2(16), np.log2(N//4), 40)).astype(int))
rng = np.random.default_rng(3)
F2sq = np.zeros(len(scales))
nrel = 40
for _ in range(nrel):
    x = E.fgn_circulant(N, H1, rng) + c*E.fgn_circulant(N, H2, rng)
    Fq = E.mfdfa_surface(x, scales, np.array([2.0]), m=2)   # F_2(s)
    F2sq += (Fq[:, 0])**2                                    # F_2^2 = mean seg variance
F2sq /= nrel
uu = np.log(scales)
g_emp = 0.5*np.log(F2sq)

# fit A1,A2 (>=0) from F2sq = A1 s^{2H1} + A2 s^{2H2} (NNLS on normalized columns)
from scipy.optimize import nnls
b0 = scales.astype(float)**(2*H1); b1 = scales.astype(float)**(2*H2)
c0, c1 = b0.max(), b1.max()
Mn = np.column_stack([b0/c0, b1/c1])
coefn, _ = nnls(Mn, F2sq)
A1f, A2f = coefn[0]/c0, coefn[1]/c1
w1n = A1f*scales**(2*H1)/(A1f*scales**(2*H1)+A2f*scales**(2*H2))
guu_closed = 2*(H1-H2)**2*w1n*(1-w1n)

# empirical G_uu via local quadratic in a sliding window on (uu, g_emp)
def local_curv(uu, g, win=9):
    out = np.full_like(g, np.nan)
    h = win//2
    for i in range(h, len(g)-h):
        cft = np.polyfit(uu[i-h:i+h+1], g[i-h:i+h+1], 2)
        out[i] = 2*cft[0]
    return out
guu_emp = local_curv(uu, g_emp)

# single-component control (exact scaling) -> G_uu ~ 0
F2sq_s = np.zeros(len(scales))
for _ in range(nrel):
    x = E.fgn_circulant(N, 0.7, rng)
    Fq = E.mfdfa_surface(x, scales, np.array([2.0]), m=2)
    F2sq_s += (Fq[:, 0])**2
F2sq_s /= nrel
guu_single = local_curv(uu, 0.5*np.log(F2sq_s))
print("\n=== NUMERIC ===")
print(f"superposition H1={H1},H2={H2}: max empirical G_uu={np.nanmax(guu_emp):.3f}, "
      f"closed-form peak={guu_closed.max():.3f}  (theory peak (Δh)^2/2={0.5*(H1-H2)**2:.3f})")
print(f"single fGn(0.7) exact scaling: mean|G_uu|={np.nanmean(np.abs(guu_single)):.4f} (≈0)")

np.savez(os.path.join(OUT, "theory_curves.npz"), log2s=uu/np.log(2), slope=np.gradient(g_emp, uu),
         guu_emp=guu_emp, guu_closed=guu_closed, guu_single=guu_single, H1=H1, H2=H2)

