# Scale curvature of the MFDFA fluctuation surface

Code, data and manuscript for

> D. Sierra-Porta, *Scale curvature of the MFDFA fluctuation surface: a calibrated diagnostic for
> crossover- and nonstationarity-induced spurious multifractality*.

Multifractal detrended fluctuation analysis (MFDFA) gives a fluctuation surface F_q(s). Written in
log-log coordinates, G(u, q) = log F_q(s) with u = log s, its scale-direction curvature
G_uu = d h_local / du vanishes under exact scaling and measures how the effective exponent drifts
with scale. The code computes |G_uu| and its q-asymmetry A, tests them against surrogates of the
same length and scale band, compares them with rank Gaussianization and a Chow-type crossover test,
and applies them to simulated processes and to sunspot, S&P 500 and ECG data.

## Requirements

Python 3 with `numpy`, `scipy`, `pandas`, `matplotlib`, `sympy` and `xlrd` (for the Shiller `.xls`
file). The manuscript is compiled with `pdflatex` and `bibtex`.

## Reproducing the results

```bash
python code/run_all.py                          # all results and figures (54 steps, a few minutes)
python code/run_all.py 07_crossover_families.py  # resume from a given script
```

`run_all.py` runs the scripts in order, writes data to `results/` and figures to `figures/`, and
fixes `PYTHONHASHSEED=0` and `OMP_NUM_THREADS=1` so that every simulation is reproducible. Scripts
can also be run one at a time from `code/`; several take a family or data-set name as argument (see
the `Usage` line at the top of each script).

To rebuild the manuscript, copy `figures/fig[1-8]_*.png` to `paper/` and compile
`paper/cas-sc-template_R1.tex`.

## Repository layout

```
code/        analysis scripts (below)
results/     generated data (CSV, NPZ, JSON)
figures/     generated figures
paper/       manuscript (Elsevier CAS template), bibliography and figures
RealDatas/   raw data
```

### Scripts

| Script | Content | Output in `results/` |
|---|---|---|
| `mfdfa_core.py` | simulators, MFDFA surface, curvature features, minimum-scale rule | (module) |
| `01_battery.py` | eleven-process battery | `features.csv`, `curves.npz` |
| `02_sweep.py` | parametric sweeps of AR(1), MRW, binomial cascades, Student-t | `sweep_features.csv` |
| `03_theory.py` | closed form of the crossover curvature: SymPy check and simulation | `theory_curves.npz` |
| `04_mrw_integral_scale.py` | MRW with varying integral scale | `mrw_Tsweep.csv` |
| `05_ecg_features.py` | features of the ECG segments | `ecg_full.csv` |
| `06_statistics.py` | bootstrap intervals and summary statistics of Sec. 5.1 and 5.6 | `uncertainty.json` |
| `07_crossover_families.py` | crossover families, sign of G_uu, G_uu(q) | `families_*.csv` |
| `08_detection_power.py` | detection power vs exponent gap and crossover scale | `power_*.csv` |
| `09_shuffle_null.py` | shuffle null for real data | `surrogate_*.csv` |
| `10_scaling_null.py` | scaling null (same length, band and h(2)) for real data | `scalingnull_real.csv` |
| `11_nonstationarity.py` | variance switching, mixtures, q-asymmetry A | `nonstat_*.csv` |
| `12_finite_size.py` | noise floors vs series length | `fsize_*.csv` |
| `13_gaussianization_chow.py` | rank Gaussianization, Chow-type test, isolated singularities, real data | `gc_*.csv` |
| `14_sunspots_min_scale.py` | sunspots and the choice of minimum scale | `sun_fixed.csv` |
| `15_real_calibration.py` | length-matched calibration for S&P 500 and ECG | `calib_*.csv` |
| `16_moment_dependence.py` | G_uu(q) and the small-dispersion approximation | `qtheory_*.csv` |
| `17_manuscript_figures.py` | manuscript figures | `figures/*.png` |
| `run_all.py` | runs everything in order | |

### Figures

| Figure | File | Content |
|---|---|---|
| 1 | `fig1_theory.png` | closed-form crossover curvature against simulation |
| 2 | `fig2_width_curvature.png` | h(q) of AR(1) and fGn; multifractal width vs signed G_uu for all families |
| 3 | `fig3_power.png` | detection power vs exponent gap and crossover scale |
| 4 | `fig4_qprofile.png` | G_uu(q) and the small-dispersion approximation |
| 5 | `fig5_probes.png` | Gaussianized width, abs(G_uu) and q-asymmetry by mechanism |
| 6 | `fig6_chow_flags.png` | Chow-type test and flag rates by probe |
| 7 | `fig7_real_data.png` | real data against length-matched nulls |
| 8 | `fig8_ecg.png` | ECG, normal sinus rhythm vs atrial premature beats |
| S1 | `figS1_finite_size.png` | finite-size noise floors (not shown in the paper) |

## Data

| Folder | Content | Source |
|---|---|---|
| `RealDatas/Sunspot/` | daily and monthly total sunspot number, version 2.0 | WDC-SILSO, Royal Observatory of Belgium, Brussels |
| `RealDatas/ECG/` | 10-s single-lead segments, classes NSR (`1 NSR`) and APB (`2 APB`) | MIT-BIH Arrhythmia Database, PhysioNet |
| `RealDatas/StockMarkets/` | `ie_data.xls` (monthly S&P composite), `SP500.csv` (daily index) | R. Shiller's online data; FRED, series SP500 |

Please cite the original sources when reusing the data.

## Notes

- Real-data analyses set the minimum scale above the longest run of constant values (`safe_smin` in
  `mfdfa_core.py`); otherwise windows with zero variance make the negative moments diverge, as in the
  daily sunspot record.
- Numerical curvatures are in base-2 units (u = log2 s, G = log2 F); the analytic results in the paper
  use natural logarithms.

## Citation

If you use this code, please cite the article above (reference to be completed on publication).
