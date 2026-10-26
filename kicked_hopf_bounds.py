# -*- coding: utf-8 -*-
"""
Numerical companion code for periodically kicked Hopf oscillators.

Usage
    python kicked_hopf_bounds.py          # quick mode (reduced grids, roughly 10-30 minutes)
    python kicked_hopf_bounds.py --full   # full grids (several hours)

Figures are written to ./figures/, text output goes to stdout.

Sections
  0  Setup and settings
  1  Reset limit: derivative range, closed-form edges, Lyapunov window, figures
  2  Thin non-degenerate rotation intervals just above the lower edge
  3  Interval-arithmetic entropy certificates (degree-zero regime)
  4  Finite relaxation: explicit edges and exact-map test
  5  Persistence in a three-dimensional test system
  6  Glucose-insulin model: normal-form reduction, window, Lyapunov exponents
  7  0-1 test for chaos
  8  Summary
"""

# %% ---------------------------------------------------------------------------------------
# 0  Setup and settings
# ------------------------------------------------------------------------------------------
import argparse, os, time, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
from numba import njit
from scipy.optimize import brentq, fsolve
from mpmath import iv, mpf, floor as mfloor, ceil as mceil

_ap = argparse.ArgumentParser(description="Kicked Hopf oscillator numerics")
_ap.add_argument("--full", action="store_true", help="use the full grids (several hours)")
QUICK = not _ap.parse_known_args()[0].full      # quick mode unless --full is given
FIG_DIR, FIG_DPI = "figures", 600
os.makedirs(FIG_DIR, exist_ok=True)
RNG = np.random.default_rng(0)
plt.rcParams.update({"font.size": 10, "mathtext.fontset": "cm"})
T0 = time.time()
def stamp(msg): print(f"[{time.time()-T0:7.1f} s] {msg}", flush=True)


# %% ---------------------------------------------------------------------------------------
# 1  Reset limit: derivative range, closed-form edges, Lyapunov window
# ------------------------------------------------------------------------------------------
def q_lo(b): return 1.0 / np.sqrt(1 + b * b)
def q_c(b):  s = np.sqrt(1 + b * b); return 0.5 * (s + np.sqrt(9 + b * b))

@njit(cache=True)
def f_reset(th, q, b, Om):
    """Reset-limit phase map."""
    re = np.cos(th) + q; im = np.sin(th)
    return Om + np.arctan2(im, re) - 0.5 * b * np.log(re * re + im * im)

@njit(cache=True)
def df_reset(th, q, b):
    return (1 + q * np.cos(th) + b * q * np.sin(th)) / (1 + q * q + 2 * q * np.cos(th))

@njit(cache=True)
def lyap_reset(q, b, Om, n_tr=1000, n=20000, th0=0.3):
    # n must be large: on quasi-periodic orbits the finite-n average converges only like 1/n
    th = th0
    for _ in range(n_tr): th = f_reset(th, q, b, Om)
    s = 0.0
    for _ in range(n):
        d = abs(df_reset(th, q, b)); s += np.log(max(d, 1e-300)); th = f_reset(th, q, b, Om)
    return s / n

# Range of f' is the Möbius image [(1-qs)/(1-q^2), (1+qs)/(1-q^2)]
th = np.linspace(-np.pi, np.pi, 200001)
for b, q in ((1.0, 0.5), (2.2, 0.8), (1.0, 1.7)):
    s = np.sqrt(1 + b * b); d = df_reset(th, q, b)
    ends = sorted([(1 - q * s) / (1 - q * q), (1 + q * s) / (1 - q * q)])
    print(f"Derivative range  b={b}, q={q}: numerical range [{d.min():.6f}, {d.max():.6f}]  vs  closed form [{ends[0]:.6f}, {ends[1]:.6f}]")

# Closed-form edges
for b in (0, 0.5, 1, 2.2, 4):
    print(f"Edges  b={b:<4}: q_lo = {q_lo(b):.4f}  (diffeomorphism below),  q_c = {q_c(b):.4f}  (contraction above)")

# Fraction of Omega with positive Lyapunov exponent
BB = np.array([0, 0.5, 1, 1.5, 2, 2.2, 3, 4])
QQ = np.geomspace(0.05, 8, 40 if QUICK else 82)
OM = np.linspace(-np.pi, np.pi, 16 if QUICK else 32, endpoint=False)
frac = np.zeros((len(BB), len(QQ)))
for i, b in enumerate(BB):
    for j, q in enumerate(QQ):
        frac[i, j] = np.mean([lyap_reset(q, b, om) > 1e-3 for om in OM])
stamp("reset-limit Lyapunov grid done")
print("\nReset limit: range of q with a positive Lyapunov exponent")
for i, b in enumerate(BB):
    pos = QQ[frac[i] > 0]
    rng_txt = f"{pos.min():.2f}–{pos.max():.2f}" if len(pos) else "none"
    print(f"  b={b:<4} q_lo={q_lo(b):.4f}  q_c={q_c(b):.4f}  lambda>0 for q in {rng_txt}")

# Reset-limit figure: geometry, phase map and derivative (b = 1)
# Panel (b,c) curves: q = 0.5 (diffeomorphism), q = 0.85 (non-invertible, degree 1),
#                     q = 1.4 (non-invertible, degree 0), q = 3.5 (contraction)
from matplotlib.gridspec import GridSpec
b1 = 1.0
fig = plt.figure(figsize=(14, 4.4))
gs = GridSpec(1, 3, width_ratios=[1.0, 1.15, 1.15], wspace=0.28)
axa, axb, axc = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])

# (a) cycle, kick and isochrons: theta - b ln R = const  (logarithmic spirals)
q_geo = 1.4
Rr = np.geomspace(0.03, 3.5, 400)
for c0 in np.linspace(-np.pi, np.pi, 9)[:-1]:
    th_ = c0 + b1 * np.log(Rr)
    axa.plot(Rr * np.cos(th_), Rr * np.sin(th_), c="0.45", lw=0.7)
tt = np.linspace(0, 2 * np.pi, 400)
axa.plot(np.cos(tt), np.sin(tt), c="mediumblue", lw=2)
axa.plot(q_geo + np.cos(tt), np.sin(tt), c="darkorange", lw=1.8, ls="--")
axa.annotate("", xy=(q_geo, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="k", lw=1.6))
axa.plot([0], [0], "k+", ms=7)
axa.text(0.55 * q_geo, 0.08, r"kick $A$", ha="center", va="bottom", fontsize=10,
         bbox=dict(fc="white", ec="none", pad=1.0, alpha=0.85))
axa.text(0, -1.22, r"cycle $r_0$", ha="center", va="top", color="mediumblue", fontsize=10)
axa.text(q_geo, 1.18, "displaced cycle", ha="center", va="bottom", color="darkorange", fontsize=10)
axa.set_xlim(-2.4, 3.1); axa.set_ylim(-2.3, 2.3); axa.set_aspect("equal"); axa.axis("off")
axa.set_title(r"(a) cycle, kick and isochrons ($b=1$)", fontsize=11)

# (b, c) phase map and derivative
t = np.linspace(-np.pi, np.pi, 4000)
curves = ((0.5,  "tab:green",  "-",  r"$q=0.5<q_{\rm lo}$: diffeomorphism"),
          (0.85, "tab:purple", "--", r"$q_{\rm lo}<q=0.85<1$: non-invertible (degree 1)"),
          (1.4,  "tab:red",    "-",  r"$q=1.4$: non-invertible (degree 0)"),
          (3.5,  "tab:blue",   "-",  r"$q=3.5>q_c$: contraction"))
for q, c, ls, lab in curves:
    axb.plot(t, np.unwrap(f_reset(t, q, b1, 0.0)), c=c, ls=ls, lw=1.8, label=lab)
    axc.plot(t, df_reset(t, q, b1), c=c, ls=ls, lw=1.8)
axb.set_title(r"(b) reset-limit phase map ($b=1$)", fontsize=11)
axb.set_xlabel(r"$\theta$"); axb.set_ylabel(r"$f(\theta)-\Omega$ (lift)")
axb.legend(fontsize=7.5, loc="upper left", framealpha=0.95)
axc.set_ylim(-3, 3)
axc.axhline(1, ls=":", c="k", lw=0.9); axc.axhline(-1, ls=":", c="k", lw=0.9); axc.axhline(0, c="k", lw=0.8)
axc.set_title(r"(c) $f'$: extreme values of the derivative", fontsize=11)
axc.set_xlabel(r"$\theta$"); axc.set_ylabel(r"$f'(\theta)$")
fig.savefig(f"{FIG_DIR}/reset_limit_maps.png", dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
plt.close(fig)

# Reset-limit window in the (q, b) plane
fig, ax = plt.subplots(figsize=(7, 4.3))
pc = ax.pcolormesh(QQ, BB, frac, cmap="Greys", shading="nearest", vmin=0, vmax=0.5)
bb = np.linspace(0, 4.2, 200)
ax.plot(q_lo(bb), bb, "b-", lw=2, label=r"$q_{\rm lo}=1/\sqrt{1+b^2}$")
ax.plot(q_c(bb), bb, "r-", lw=2, label=r"$q_c=(\sqrt{1+b^2}+\sqrt{9+b^2})/2$")
ax.set_xscale("log"); ax.set_xlabel(r"$q=A/r_0$"); ax.set_ylabel(r"twist $b$"); ax.legend(fontsize=8, loc="lower left")
plt.colorbar(pc, label=r"fraction of $\Omega$ with $\lambda>0$"); plt.tight_layout()
fig.savefig(f"{FIG_DIR}/reset_limit_window.png", dpi=FIG_DPI, bbox_inches="tight", facecolor="white"); plt.close(fig)


# %% ---------------------------------------------------------------------------------------
# 2  Water functions and thin rotation intervals above the lower edge
# ------------------------------------------------------------------------------------------
TP = 2 * np.pi
def lift_setup(b, q):
    s = np.sqrt(1 + b * b); chi = np.arctan(b); a = np.arccos(-1 / (q * s))
    thM, thm = chi + a, chi + TP - a
    F = lambda x: x + np.angle(1 + q * np.exp(-1j * x)) - b * np.log(abs(1 + q * np.exp(-1j * x)))
    dF = lambda x: (1 + q * s * np.cos(x - chi)) / (1 + q * q + 2 * q * np.cos(x))
    if dF(0.5 * (thM + thm)) > 0: thM, thm = thm - TP, thM
    zu = brentq(lambda x: F(x) - F(thM), thm, thM + TP)
    zl = brentq(lambda x: F(x) - F(thm), thm - TP, thM)
    return thM, thm, zu, zl, F(thM), F(thm)

@njit(cache=True)
def _Fn(x, b, q):
    z = 1 + q * np.exp(-1j * x); return x + np.angle(z) - b * np.log(abs(z))

@njit(cache=True)
def rot_water(kind, Om, b, q, thM, thm, zu, zl, FM, Fm, N):
    """Rotation number of the upper (kind 0) or lower (kind 1) water function of F + Om."""
    x = 0.0
    for _ in range(N):
        if kind == 0:
            k = np.floor((x - thM) / TP); y = x - k * TP
            v = FM if y <= zu else _Fn(y, b, q)
        else:
            k = np.floor((x - zl) / TP); y = x - k * TP
            v = Fm if y <= thm else _Fn(y, b, q)
        x = v + k * TP + Om
    return x / (TP * N)

golden = (np.sqrt(5) - 1) / 2
N_rot = 400_000 if QUICK else 4_000_000
print("\nThin rotation intervals (b = 1): rotation interval at the phase where its right end is the golden mean")
for dq in (1e-1, 1e-2, 1e-3):
    q = q_lo(1.0) + dq; P = lift_setup(1.0, q)
    g = lambda Om: rot_water(0, Om, 1.0, q, *P, N_rot) - golden
    Oms = np.linspace(0, TP, 200); v = [g(o) for o in Oms]
    i = next(i for i in range(199) if v[i] < 0 <= v[i + 1])
    Om = brentq(g, Oms[i], Oms[i + 1], xtol=1e-12)
    ru = rot_water(0, Om, 1.0, q, *P, 4 * N_rot); rl = rot_water(1, Om, 1.0, q, *P, 4 * N_rot)
    print(f"  q - q_lo = {dq:g}: rho_upper = {ru:.8f}, rho_lower = {rl:.8f}, width = {ru-rl:.2e}  (non-degenerate => h_top > 0)")
stamp("rotation-interval section done")


# %% ---------------------------------------------------------------------------------------
# 3  Computer-assisted entropy certificates (degree zero, 40-digit intervals)
# ------------------------------------------------------------------------------------------
iv.dps = 40
NOM = 384
def F0_np(x, q, b, Om):
    re = np.cos(x) + q; im = np.sin(x); return Om + np.arctan2(im, re) - 0.5 * b * np.log(re * re + im * im)
def F0_iv(x, q, b, Om):
    re = iv.cos(x) + q; im = iv.sin(x); return Om + iv.atan2(im, re) - b / 2 * iv.log(re ** 2 + im ** 2)
def it_iv(x, n, q, b, Om):
    for _ in range(n): x = F0_iv(x, q, b, Om)
    return x
def Om_iv(k): return -iv.pi + 2 * iv.pi * k / NOM
def Om_f(k):  return -np.pi + 2 * np.pi * k / NOM

def arc_ok(qi, bi, M):
    """f(S^1) is a proper arc: union of interval enclosures has length < 2*pi (independent of Omega)."""
    lo = hi = None; e = [iv.mpf(0) + 2 * iv.pi * j / M for j in range(M + 1)]
    for j in range(M):
        Y = F0_iv(iv.mpf([e[j].a, e[j + 1].b]), qi, bi, iv.mpf(0))
        lo = Y.a if lo is None or Y.a < lo else lo; hi = Y.b if hi is None or Y.b > hi else hi
    return (iv.mpf(hi) - iv.mpf(lo)).b < (2 * iv.pi).a

def nofix(a, c, qi, bi, Omi, pieces=64):
    ed = np.linspace(a, c, pieces + 1)
    for u, v in zip(ed[:-1], ed[1:]):
        X = iv.mpf([u, v]); G = (F0_iv(X, qi, bi, Omi) - X) / (2 * iv.pi)
        if mfloor(G.b) >= mceil(G.a): return False
    return True

def cert_deg0(qs, bs, M=4000, nx=3000, nlist=(3, 5, 7, 11, 13)):
    """True if some Omega_k gives a proper-arc map with a periodic orbit of odd prime period n => h_top > 0."""
    q, b = float(qs), float(bs); qi, bi = iv.mpf(qs), iv.mpf(bs)
    if not arc_ok(qi, bi, M): return False, None
    xs = np.linspace(0, TP, nx, endpoint=False)
    for k in range(0, NOM, 2):
        Om = Om_f(k); Omi = Om_iv(k)
        for n in nlist:
            y = xs.copy()
            for _ in range(n): y = F0_np(y, q, b, Om)
            gn = y - xs; kk_all = np.round(gn / TP)
            for kk in np.unique(kk_all):
                g = gn - kk * TP
                idx = np.where((np.sign(g[:-1]) != np.sign(g[1:])) & (np.abs(g[:-1]) < 1) & (kk_all[:-1] == kk))[0]
                for c in idx[:6]:
                    a_, c_ = xs[c], xs[c + 1]
                    ha = it_iv(iv.mpf(a_), n, qi, bi, Omi) - iv.mpf(a_) - 2 * int(kk) * iv.pi
                    hb = it_iv(iv.mpf(c_), n, qi, bi, Omi) - iv.mpf(c_) - 2 * int(kk) * iv.pi
                    if ((ha.b < 0 and hb.a > 0) or (hb.b < 0 and ha.a > 0)) and nofix(a_, c_, qi, bi, Omi):
                        return True, (k, n)
    return False, None

print("\nDegree-zero certificates (40-digit interval arithmetic)")
cases = [("1", "1.2000"), ("2.2", "1.5000")] if QUICK else [(bs, f"{q:.4f}") for bs in ("0", "1", "2.2", "4")
                                                            for q in np.linspace(1.01, q_c(float(bs)) - 0.005, 24)]
for bs, qs in cases:
    ok, info = cert_deg0(qs, bs, M=2000 if QUICK else 4000)
    print(f"  b={bs}, q={qs}: certified={ok}" + (f"  (Omega_k with k={info[0]}, odd prime period {info[1]})" if ok else ""))
stamp("certificates done")


# %% ---------------------------------------------------------------------------------------
# 4  Finite relaxation: explicit edges and exact-map test
# ------------------------------------------------------------------------------------------
def h_rad(x, E):  return x / np.sqrt(x * x * (1 - E) + E)
def hp_rad(x, E): return E / (x * x * (1 - E) + E) ** 1.5

def sink_ok(q, b, tau):
    """Global-sink condition. Returns (holds, contraction factor Lambda)."""
    E = np.exp(-2 * tau); s = np.sqrt(1 + b * b); ro = 1 / np.sqrt(1 - E)
    if q <= ro: return False, np.inf
    qh = q / ro; l1 = (1 + qh * s) / (qh * qh - 1)
    if l1 >= 1: return False, np.inf
    Rs = q - ro; cr = s / Rs; er = hp_rad(Rs, E); et = er * ro * q / Rs
    if cr * et >= (1 - l1) * (1 - er): return False, np.inf
    if et == 0: return True, l1
    eta = (-(er - l1) + np.sqrt((er - l1) ** 2 + 4 * et * cr)) / (2 * et)
    return True, l1 + cr / eta

def rho_in(q, E):
    g = lambda r: h_rad(r - q, E) - r
    rs = np.linspace(1.0 - 1e-12, q + 1e-9, 4000); v = g(rs); idx = np.where(v >= 0)[0]
    if len(idx) == 0: return np.nan
    j = idx[0]
    return rs[0] if j == 0 else brentq(g, rs[j], rs[j - 1])

def circle_ok(q, b, tau):
    """Attracting-invariant-circle condition, with the optimal gamma_-."""
    if q >= 1: return False, np.nan
    E = np.exp(-2 * tau); s = np.sqrt(1 + b * b); ro = 1 / np.sqrt(1 - E); ri = rho_in(q, E)
    if not np.isfinite(ri) or ri <= q: return False, np.nan
    qv = q / ri
    if qv * s >= 1: return False, np.nan
    m = (1 - qv * s) / (1 - qv * qv); Rs = ri - q; cr = s / Rs; er = hp_rad(Rs, E); et = er * ro * q / Rs
    disc = (m - er) ** 2 - 4 * cr * et
    if disc < 0 or m - er <= 0: return False, np.inf
    gam = 2 * et / ((m - er) + np.sqrt(disc))
    if not (gam >= 0 and cr * gam < m): return False, np.inf
    LG = (et + er * gam) * cr / (m - cr * gam) + er
    return LG < 1, LG

def Q_hi(b, tau):
    lo, hi = 1.0, 60.0
    if not sink_ok(hi, b, tau)[0]: return np.nan
    for _ in range(80):
        m = 0.5 * (lo + hi)
        if sink_ok(m, b, tau)[0]: hi = m
        else: lo = m
    return hi

def Q_lo(b, tau):
    lo, hi = 1e-9, 1.0 - 1e-12
    if not circle_ok(lo, b, tau)[0]: return np.nan
    for _ in range(60):
        m = 0.5 * (lo + hi)
        if circle_ok(m, b, tau)[0]: lo = m
        else: hi = m
    return lo

print("\nExplicit finite-tau edges Q_lo / Q_hi (units of r0)")
TAUS = (1, 2, 3, 5, 8, 12)
print("   b   " + "  ".join(f"tau={t:<9}" for t in TAUS) + "  tau->inf")
for b in (0, 1, 2.2, 4):
    print(f"  {b:<4} " + "  ".join(f"{Q_lo(b,t):.3f}/{Q_hi(b,t):.3f}" for t in TAUS) + f"  {q_lo(b):.3f}/{q_c(b):.3f}")

@njit(cache=True)
def P_exact(rho, th, q, b, tau, Om):
    """Kicked map, exact (closed-form flow)."""
    E = np.exp(-2 * tau)
    wr = rho * np.cos(th) + q; wi = rho * np.sin(th); R2 = wr * wr + wi * wi; X = R2 * (1 - E) + E
    return np.sqrt(R2) / np.sqrt(X), Om + np.arctan2(wi, wr) - 0.5 * b * np.log(X)

@njit(cache=True)
def lyap_exact(q, b, tau, Om, n_tr=1500, n=12000):
    z = 0.9 + 0.1j; v = 1.0 + 0.0j; s = 0.0; eps = 1e-8
    for k in range(n_tr + n):
        r, t = P_exact(abs(z), np.angle(z), q, b, tau, Om); z1 = r * np.exp(1j * t)
        zp = z + eps * v; rp, tp = P_exact(abs(zp), np.angle(zp), q, b, tau, Om); d = (rp * np.exp(1j * tp) - z1) / eps
        nd = abs(d)
        if k >= n_tr: s += np.log(max(nd, 1e-300))
        v = d / max(nd, 1e-300); z = z1
    return s / n

print("\nExact-map test (no chaos where the sink or circle condition holds)")
nq = 40 if QUICK else 140; taus_t = (0.5, 1, 2, 5, 12) if QUICK else tuple(np.geomspace(0.5, 12, 14))
fig, axs = plt.subplots(2, 2, figsize=(9, 6.5)); axs = axs.ravel()
viol = 0; ncirc = 0; nsink = 0
for kb, b in enumerate((0.0, 1.0, 2.2, 4.0)):
    a = axs[kb]; tt = np.geomspace(0.3, 20, 40)
    a.plot([Q_lo(b, x) for x in tt], tt, "b-", label=r"$Q_{\rm lo}$"); a.plot([Q_hi(b, x) for x in tt], tt, "r-", label=r"$Q_{\rm hi}$")
    a.axvline(q_lo(b), c="b", ls=":"); a.axvline(q_c(b), c="r", ls=":")
    for tau in taus_t:
        for q in np.geomspace(0.02, 5, nq):
            c_ok = circle_ok(q, b, tau)[0]; s_ok = sink_ok(q, b, tau)[0]
            lam = max(lyap_exact(q, b, tau, om) for om in np.linspace(-np.pi, np.pi, 4 if QUICK else 8, endpoint=False))
            ncirc += c_ok; nsink += s_ok
            if lam > 1e-3:
                a.plot(q, tau, "k.", ms=3)
                if c_ok or s_ok: viol += 1
    a.set_xscale("log"); a.set_yscale("log"); a.set_title(f"$b={b:g}$"); a.set_xlabel(r"$q$"); a.set_ylabel(r"$\tau$")
axs[0].legend(fontsize=8); plt.tight_layout()
fig.savefig(f"{FIG_DIR}/finite_relaxation_test.png", dpi=FIG_DPI, bbox_inches="tight", facecolor="white"); plt.close(fig)
print(f"  cases satisfying the circle condition: {ncirc}, the sink condition: {nsink}, chaotic among them: {viol}")
stamp("finite-tau section done")


# %% ---------------------------------------------------------------------------------------
# 5  Persistence in a three-dimensional test system
# ------------------------------------------------------------------------------------------
w_, lam_, c_, b0_, e_ = 1.0, 1.0, 0.5, 0.5, 0.3      # z' = (eps+i)z-(1+0.5i)|z|^2 z + 0.5 z y + 0.3 conj(z)|z|^2,  y' = -y+|z|^2
alpha_ = 1 - c_ / lam_; b_eff = b0_ / alpha_        # effective cubic coefficient -(0.5+0.5i): b = 1
kc, ks = 1.0, 0.7

@njit(cache=True)
def _rhs3(zr, zi, y, eps):
    z = zr + 1j * zi; a2 = zr * zr + zi * zi
    dz = (eps + 1j * w_) * z - (1 + 1j * b0_) * a2 * z + c_ * z * y + e_ * np.conj(z) * a2
    return dz.real, dz.imag, -lam_ * y + a2

@njit(cache=True)
def _flow3(zr, zi, y, eps, T, dt):
    n = int(np.ceil(T / dt)); h = T / n
    for _ in range(n):
        k1 = _rhs3(zr, zi, y, eps); k2 = _rhs3(zr + .5*h*k1[0], zi + .5*h*k1[1], y + .5*h*k1[2], eps)
        k3 = _rhs3(zr + .5*h*k2[0], zi + .5*h*k2[1], y + .5*h*k2[2], eps); k4 = _rhs3(zr + h*k3[0], zi + h*k3[1], y + h*k3[2], eps)
        zr += h/6*(k1[0]+2*k2[0]+2*k3[0]+k4[0]); zi += h/6*(k1[1]+2*k2[1]+2*k3[1]+k4[1]); y += h/6*(k1[2]+2*k2[2]+2*k3[2]+k4[2])
    return zr, zi, y

def kicked3(zeta, yh, eps, q, tau):
    r = np.sqrt(eps / alpha_); a = q * r / kc
    zr, zi, yy = _flow3((r*zeta + a*kc).real, (r*zeta + a*kc).imag, r*yh + a*ks, eps, tau/eps, 0.02)
    return (zr + 1j*zi) / r, yy / r

def model3(zeta, eps, q, tau):
    E = np.exp(-2*tau); W = zeta + q; R = abs(W); X = R*R*(1-E) + E
    return R/np.sqrt(X) * np.exp(1j*(np.angle(W) + (w_/eps - b_eff)*tau - 0.5*b_eff*np.log(X)))

print("\nDifference between the scaled kicked map and the model map")
for eps in ((0.04, 0.01) if QUICK else (0.04, 0.01, 0.0025)):
    err = max(abs(kicked3(rho*np.exp(1j*t), 0.0, eps, 0.5, 2.0)[0] - model3(rho*np.exp(1j*t), eps, 0.5, 2.0))
              for t in np.linspace(0, TP, 12, endpoint=False) for rho in (0.8, 1.0))
    print(f"  eps={eps:<7} r0={np.sqrt(eps/alpha_):.3f}  max difference={err:.4f}  ratio to r0={err/np.sqrt(eps/alpha_):.3f}")
qs_ = 1.15 * Q_hi(b_eff, 3.0); finals = []
for k in range(4):
    z, yh = RNG.uniform(-1.2, 1.2) + 1j*RNG.uniform(-1.2, 1.2), RNG.uniform(-1, 1)
    for _ in range(40): z, yh = kicked3(z, yh, 0.01, qs_, 3.0)
    finals.append(z)
print(f"  sink regime q={qs_:.3f}, tau=3: spread of 4 random initial states after 40 kicks = {np.ptp(np.abs(finals)):.1e}")
stamp("three-dimensional test done")


# %% ---------------------------------------------------------------------------------------
# 6  Glucose–insulin model
# ------------------------------------------------------------------------------------------
# parameter vector: Vp Vi Vg E tp ti td Rm a1 C1 C2 C3 C4 C5 Ub U0 Um Rg alpha beta
P_DEF = dict(Vp=3.0, Vi=11.0, Vg=10.0, E=0.2, tp=6.0, ti=100.0, td=36.0, Rm=209.0, a1=6.67, C1=300.0, C2=144.0,
             C3=100.0, C4=80.0, C5=26.0, Ub=72.0, U0=4.0, Um=94.0, Rg=180.0, alpha=7.5, beta=1.77)
NAMES = list(P_DEF)
C4_MINUS = 80.0 * (1/11.0 + 1/(0.2*100.0)) / (1/11.0 - 1/(0.2*100.0))   # "−" convention (= 275.6)

def p_endo(e, minus=False):
    """Endocrine point e = (s_I, r_m, r_g, u_b, t_d, G_m) -> parameter vector."""
    s, rm, rg, ub, td, Gm = e
    d = dict(P_DEF); d.update(td=td, Rg=180.0*rg, Ub=72.0*ub, Um=94.0*s, U0=4.0*s, Rm=209.0*rm)
    if minus: d["C4"] = C4_MINUS
    return np.array([d[k] for k in NAMES])

@njit(cache=True)
def rhs6(x, p, g):
    Vp, Vi, Vg, E, tp, ti, td = p[0], p[1], p[2], p[3], p[4], p[5], p[6]
    Rm, a1, C1, C2, C3, C4, C5 = p[7], p[8], p[9], p[10], p[11], p[12], p[13]
    Ub, U0, Um, Rg, al, be = p[14], p[15], p[16], p[17], p[18], p[19]
    Ip, Ii, G, h3 = x[0], x[1], x[2], x[5]; Iiq = Ii if Ii > 1e-12 else 1e-12
    f1 = Rm / (1 + np.exp(-G/(Vg*C1) + a1)); f2 = Ub*(1 - np.exp(-G/(C2*Vg))); f3 = G/(C3*Vg)
    kap = (1/C4)*(1/Vi + 1/(E*ti)); f4 = U0 + (Um - U0)/(1 + (kap*Iiq)**(-be)); f5 = Rg/(1 + np.exp(al*(h3/(C5*Vp) - 1)))
    fl = E*(Ip/Vp - Ii/Vi); k3 = 3.0/td; dx = np.empty(6)
    dx[0] = f1 - fl - Ip/tp; dx[1] = fl - Ii/ti; dx[2] = g - f2 - f3*f4 + f5
    dx[3] = k3*(Ip - x[3]); dx[4] = k3*(x[3] - x[4]); dx[5] = k3*(x[4] - x[5])
    return dx

@njit(cache=True)
def jac6(x, p):
    Vp, Vi, Vg, E, tp, ti, td = p[0], p[1], p[2], p[3], p[4], p[5], p[6]
    Rm, a1, C1, C2, C3, C4, C5 = p[7], p[8], p[9], p[10], p[11], p[12], p[13]
    Ub, U0, Um, Rg, al, be = p[14], p[15], p[16], p[17], p[18], p[19]
    Ii, G, h3 = x[1], x[2], x[5]; Iiq = Ii if Ii > 1e-12 else 1e-12
    e1 = np.exp(-G/(Vg*C1) + a1); df1 = Rm*e1/(1 + e1)**2/(Vg*C1); df2 = Ub/(C2*Vg)*np.exp(-G/(C2*Vg))
    f3 = G/(C3*Vg); df3 = 1/(C3*Vg); kap = (1/C4)*(1/Vi + 1/(E*ti)); qq = (kap*Iiq)**(-be)
    f4 = U0 + (Um - U0)/(1 + qq); df4 = (Um - U0)*be*qq/(Iiq*(1 + qq)**2)
    e5 = np.exp(al*(h3/(C5*Vp) - 1)); df5 = -Rg*al/(C5*Vp)*e5/(1 + e5)**2; k3 = 3.0/td
    J = np.zeros((6, 6))
    J[0, 0] = -E/Vp - 1/tp; J[0, 1] = E/Vi; J[0, 2] = df1; J[1, 0] = E/Vp; J[1, 1] = -E/Vi - 1/ti
    J[2, 1] = -f3*df4; J[2, 2] = -df2 - df3*f4; J[2, 5] = df5
    J[3, 0] = k3; J[3, 3] = -k3; J[4, 3] = k3; J[4, 4] = -k3; J[5, 4] = k3; J[5, 5] = -k3
    return J

@njit(cache=True)
def rk4(x, p, g, dt):
    k1 = rhs6(x, p, g); k2 = rhs6(x + .5*dt*k1, p, g); k3 = rhs6(x + .5*dt*k2, p, g); k4 = rhs6(x + dt*k3, p, g)
    return x + dt/6*(k1 + 2*k2 + 2*k3 + k4)

@njit(cache=True)
def rk4_tan(x, v, p, g, dt):
    k1 = rhs6(x, p, g); l1 = jac6(x, p) @ v
    x2 = x + .5*dt*k1; v2 = v + .5*dt*l1; k2 = rhs6(x2, p, g); l2 = jac6(x2, p) @ v2
    x3 = x + .5*dt*k2; v3 = v + .5*dt*l2; k3 = rhs6(x3, p, g); l3 = jac6(x3, p) @ v3
    x4 = x + dt*k3; v4 = v + dt*l3; k4 = rhs6(x4, p, g); l4 = jac6(x4, p) @ v4
    return x + dt/6*(k1 + 2*k2 + 2*k3 + k4), v + dt/6*(l1 + 2*l2 + 2*l3 + l4)

X0 = np.array([100.0, 100.0, 10000.0, 100.0, 100.0, 100.0])

@njit(cache=True)
def lyap_kicked(p, G0, kick, T, dt, n_tr_kicks, n_kicks, x0, v0):
    """Largest Lyapunov exponent [1/min] of the kicked model (periodic kicks G -> G + kick every T min)."""
    ns = int(round(T / dt)); h = T / ns; x = x0.copy()
    for _ in range(n_tr_kicks):
        for _ in range(ns): x = rk4(x, p, G0, h)
        x[2] += kick
    v = v0 / np.sqrt(np.sum(v0*v0)); s = 0.0; s1 = 0.0
    for k in range(n_kicks):
        for _ in range(ns): x, v = rk4_tan(x, v, p, G0, h)
        x[2] += kick
        nv = np.sqrt(np.sum(v*v)); s += np.log(nv); v = v / nv
        if k == n_kicks // 2 - 1: s1 = s
    Tm = n_kicks * T
    return s / Tm, s1 / (0.5*Tm), (s - s1) / (0.5*Tm)

@njit(cache=True)
def strobe(p, G0, kick, T, dt, n_tr, N, x0):
    ns = int(round(T / dt)); h = T / ns; x = x0.copy(); out = np.empty(N)
    for k in range(n_tr + N):
        for _ in range(ns): x = rk4(x, p, G0, h)
        if k >= n_tr: out[k - n_tr] = x[2]
        x[2] += kick
    return out

def equilibrium(p, G0):
    x = X0.copy()
    for _ in range(3000): x = rk4(x, p, G0, 1.0)
    return fsolve(lambda z: rhs6(np.asarray(z, float), p, G0), x, xtol=1e-12)

def normal_form(p, G0, kick_mg, hfd=2e-3):
    """Critical eigenvalue mu+i*omega, cubic coefficient l1 = -(alpha+i*beta) (standard Hopf normal-form formula), projected kick."""
    xe = equilibrium(p, G0); S = np.maximum(np.abs(xe), 1e-6)
    J = jac6(xe, p) * S[None, :] / S[:, None]
    lam, V = np.linalg.eig(J); k = int(np.argmax(lam.real + 1e-9*np.sign(lam.imag)))
    if lam[k].imag < 0: k = int(np.argmin(np.abs(lam - np.conj(lam[k]))))
    mu, om = lam[k].real, lam[k].imag; qv = V[:, k] / np.linalg.norm(V[:, k])
    lL, W = np.linalg.eig(J.T); pv = W[:, int(np.argmin(np.abs(lL - np.conj(lam[k]))))]; pv = pv / np.conj(np.vdot(pv, qv))
    f = lambda y: rhs6(xe + S*y, p, G0) / S
    B = lambda u, v: (f(hfd*(u+v)) - f(hfd*(u-v)) - f(hfd*(-u+v)) + f(-hfd*(u+v))) / (4*hfd*hfd)
    def C(u, v, w):
        s = np.zeros(6)
        for a in (1, -1):
            for b in (1, -1):
                for c in (1, -1): s += a*b*c*f(hfd*(a*u + b*v + c*w))
        return s / (8*hfd**3)
    Bc = lambda u, v: (B(u.real, v.real) - B(u.imag, v.imag)) + 1j*(B(u.real, v.imag) + B(u.imag, v.real))
    def Cc(u, v, w):
        out = np.zeros(6, complex)
        for cu, uu in ((1, u.real), (1j, u.imag)):
            for cv, vv in ((1, v.real), (1j, v.imag)):
                for cw, ww in ((1, w.real), (1j, w.imag)):
                    if np.any(uu) and np.any(vv) and np.any(ww): out += cu*cv*cw*C(uu, vv, ww)
        return out
    l1 = (0.5*np.vdot(pv, Cc(qv, qv, np.conj(qv))) - np.vdot(pv, Bc(qv, np.linalg.solve(J, Bc(qv, np.conj(qv)))))
          + 0.5*np.vdot(pv, Bc(np.conj(qv), np.linalg.solve(2j*om*np.eye(6) - J, Bc(qv, qv)))))
    dx = np.zeros(6); dx[2] = kick_mg / S[2]
    return dict(mu=mu, omega=om, alpha=-l1.real, beta=-l1.imag, dz=abs(np.vdot(pv, dx)))

def analyse_run(e, A, T, minus):
    """Normal-form reduction at the between-kick equilibrium and position relative to the window."""
    p = p_endo(e, minus); kick = A * 10.0 * p[2]; G0 = e[5] - kick / T
    nf = normal_form(p, G0, kick)
    out = dict(p=p, G0=G0, kick=kick, **nf)
    if nf["mu"] > 0 and nf["alpha"] > 0:
        r0 = np.sqrt(nf["mu"] / nf["alpha"]); q0 = nf["dz"] / r0; b0 = abs(nf["beta"] / nf["alpha"]); tau0 = nf["mu"] * T
        inside = (not circle_ok(q0, b0, tau0)[0]) and (not sink_ok(q0, b0, tau0)[0])
        out.update(q0=q0, b0=b0, tau0=tau0, inside=inside)
    return out

# Example parameter sets (chaotic and regular cases): e = (s_I, r_m, r_g, u_b, t_d [min], G_m [mg/min]), A [mg/dL], T [min]
RUNS = [
    ("+", "chaotic",  [1.986476295, 0.738998903, 2.746897523, 0.7355442572, 51.86156073, 69.1518589], 89.09859296, 279.8768427),
    ("+", "chaotic",  [1.047801637, 0.731720497, 2.120168497, 1.420596651, 58.70617944, 75.15790827], 110.9516802, 524.6058137),
    ("+", "regular",  [1.694633205, 1.106376227, 1.840633472, 0.6158936158, 44.01457717, 300.8676992], 53.34740873, 432.051638),
    ("−", "chaotic",  [0.4568600289, 0.2269505671, 2.908298643, 1.305850719, 52.92667368, 97.90416288], 31.88131294, 715.0723339),
    ("−", "chaotic",  [0.6528438004, 1.06034324, 2.98880188, 0.7727636047, 42.85055084, 79.99082652], 46.5897856, 449.7297645),
    ("−", "regular",  [0.3202570984, 0.902601194, 2.250043996, 0.977577424, 29.80760597, 71.32915895], 83.78527072, 704.6489371),
]
n_meas_min = 2e5 if QUICK else 4e5
print("\nGlucose–insulin model: normal-form reduction, window and Lyapunov exponents")
results = []
for conv, label, e, A, T in RUNS:
    r = analyse_run(np.array(e), A, T, conv == "−")
    lams = [lyap_kicked(r["p"], r["G0"], r["kick"], T, 0.1, max(10, int(5000/T)), max(20, int(n_meas_min/T)), X0*sc,
                        np.random.default_rng(17*i).standard_normal(6))[0] for i, sc in enumerate((1.0, 1.3, 0.7))]
    lam = float(np.median(lams)); results.append((conv, label, r, lam, A, T))
    print(f"  [{conv}] {label:8s} A={A:6.1f} mg/dL T={T:5.0f} min | mu0={r['mu']:+.2e} b0={r.get('b0',np.nan):.2f} "
          f"q0={r.get('q0',np.nan):.2f} tau0={r.get('tau0',np.nan):.2f} inside window: {r.get('inside','n/a')} | "
          f"lambda (median of 3 ICs) = {lam:+.2e} /min  ->  {'chaotic' if lam > 3e-4 else 'not chaotic'}")
stamp("model section done")

# Optional mini-scan: random endocrine points and kicks, chaos must lie inside the window
N_SCAN = 40 if QUICK else 400
lo = np.array([0.25, 0.15, 0.5, 0.6, 15, 60, 5, 60]); hi = np.array([2, 2, 3, 1.6, 60, 450, 120, 720])
scan = []
for i in range(N_SCAN):
    x = lo + (hi - lo) * RNG.random(8); e, A, T = x[:6], x[6], x[7]
    if e[5] - A*10*10/T < 0: continue
    try:
        r = analyse_run(e, A, T, False)
    except Exception:
        continue
    if "q0" not in r: continue
    lam = lyap_kicked(r["p"], r["G0"], r["kick"], T, 0.2, max(5, int(5000/T)), max(10, int(2e4/T)), X0, np.ones(6))[0]
    scan.append((r["q0"], r["b0"], r["tau0"], r["inside"], lam))
scan = np.array(scan, dtype=float)
if len(scan):
    ch = scan[:, 4] > 3e-4
    print(f"\nMini-scan ({len(scan)} runs with a between-kick cycle, screening criterion): {int(ch.sum())} chaotic, "
          f"{int(np.sum(ch & (scan[:,3] == 1)))} of them inside the window; window admits {100*np.mean(scan[:,3]==1):.0f}% of runs")
stamp("mini-scan done")


# %% ---------------------------------------------------------------------------------------
# 7  0–1 test for chaos (correlation method)
# ------------------------------------------------------------------------------------------
def K01(phi, nc=100, seed=0):
    rng = np.random.default_rng(seed); N = len(phi); ncut = N // 10
    if phi.std() <= 1e-9 * max(abs(phi.mean()), 1.0): return 0.0          # 1:1 locking
    phi = (phi - phi.mean()) / phi.std(); j = np.arange(1, N + 1); n = np.arange(1, ncut + 1); Ks = []
    for c in rng.uniform(np.pi/5, 4*np.pi/5, nc):
        pc = np.cumsum(phi*np.cos(j*c)); qc = np.cumsum(phi*np.sin(j*c))
        M = np.array([np.mean((pc[m:] - pc[:-m])**2 + (qc[m:] - qc[:-m])**2) for m in n])
        D = M - phi.mean()**2 * (1 - np.cos(n*c)) / (1 - np.cos(c))
        Ks.append(np.corrcoef(n, D)[0, 1])
    return float(np.median(Ks))

print("\n0–1 test on the stroboscopic glucose series of the example runs (K near 1: chaos, near 0: regular)")
for conv, label, r, lam, A, T in results:
    g = strobe(r["p"], r["G0"], r["kick"], T, 0.2, max(50, int(np.ceil(1e4/T))), 1000 if QUICK else 2000, X0)
    print(f"  [{conv}] {label:8s} lambda={lam:+.2e}  K={K01(g):.3f}")
stamp("0–1 test done")


# %% ---------------------------------------------------------------------------------------
# 8  Summary
# ------------------------------------------------------------------------------------------
print("""
Summary
  * Reset limit: diffeomorphism iff q < 1/sqrt(1+b^2); contraction iff q > (sqrt(1+b^2)+sqrt(9+b^2))/2.
  * Lower edge sharp: just above it the rotation interval is non-degenerate (Section 2 above).
  * Finite relaxation: explicit edges Q_lo, Q_hi converge to the reset-limit edges; no exact-map chaos where
    the sink or circle condition holds (Section 4 above).
  * Glucose–insulin model: chaotic runs lie inside the window predicted from the normal form at their own
    between-kick equilibrium (Section 6 above).
Run with --full to use the full grids.""")
stamp("finished")
