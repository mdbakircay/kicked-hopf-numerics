# Kicked Hopf oscillator numerics

Python code for numerical experiments on periodically kicked Hopf oscillators:
the reset-limit phase map, its derivative range and rotation intervals,
interval-arithmetic certificates of positive topological entropy, the finite-relaxation
regime, a three-dimensional test system, and a normal-form analysis of a
glucose-insulin model with Lyapunov exponents and the 0-1 test.

Everything is in a single script, `kicked_hopf_bounds.py`, organised in sections that
are run from top to bottom.

## Contents

| File | Purpose |
|---|---|
| `kicked_hopf_bounds.py` | All computations and figures |
| `requirements.txt` | Python dependencies |
| `results/run_quick_log.txt` | Console output of a quick-mode run (reference) |
| `LICENSE` | MIT licence |

## Sections of the script

| # | Content |
|---|---|
| 0 | Setup and settings |
| 1 | Reset limit: derivative range, closed-form edges `q_lo(b)` and `q_c(b)`, Lyapunov window, figures |
| 2 | Thin non-degenerate rotation intervals just above the lower edge (water functions) |
| 3 | Interval-arithmetic entropy certificates, degree-zero regime (40-digit intervals) |
| 4 | Finite relaxation: explicit edges `Q_lo`, `Q_hi` and an exact-map test |
| 5 | Persistence in a three-dimensional test system |
| 6 | Glucose-insulin model: normal-form reduction, window test, Lyapunov exponents |
| 7 | 0-1 test for chaos |
| 8 | Summary |

## Requirements

Python 3.10 or newer (tested with Python 3.12). Install the dependencies with

```bash
pip install -r requirements.txt
```

The reference output was produced with NumPy 2.4, SciPy 1.17, Matplotlib 3.10,
Numba 0.68 and mpmath 1.3.

## Usage

```bash
python kicked_hopf_bounds.py          # quick mode: reduced grids
python kicked_hopf_bounds.py --full   # full grids: several hours
```

Quick mode finished in about 3 minutes on a single CPU core (the first run is slower
because Numba compiles and caches the numerical kernels). Full mode uses finer grids
and longer orbits and takes several hours.

Console output is printed to standard output; to keep it, run for example

```bash
python kicked_hopf_bounds.py | tee run_log.txt
```

## Outputs

The script creates a `figures/` folder (600 dpi PNG files):

- `reset_limit_maps.png`: geometry of the reset limit, phase maps and derivatives for `b = 1`
- `reset_limit_window.png`: fraction of phases with a positive Lyapunov exponent in the `(q, b)` plane, with the closed-form edges
- `finite_relaxation_test.png`: explicit edges and exact-map test for several twist values

`results/run_quick_log.txt` is the console output of a quick-mode run and can be used to
check a local installation. Full mode uses finer grids, so its numbers may differ from
the quick-mode log in the last digits.

## Reproducibility

- Random numbers use fixed seeds, so repeated runs give the same output on the same
  software versions.
- Floating-point results can differ slightly across platforms and library versions.
- Numba creates cache files in `__pycache__/`; they are safe to delete.

## Licence

Released under the MIT licence (see `LICENSE`).
