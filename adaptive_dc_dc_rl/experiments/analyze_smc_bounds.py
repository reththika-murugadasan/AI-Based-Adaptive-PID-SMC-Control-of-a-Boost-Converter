"""Safe parameter-range analysis for the adaptive SMC (Stage 7).

For a candidate box of multiplicative factors around the validated SMC parameters
(Kv, Ki, K, phi) every corner (and optionally mid-points) of the box is simulated
closed-loop from a cold start on the extreme operating points
(Vin 10/14 V x R 10/30 ohm x Vref 20/28 V + nominal) and judged by:
  Each plant is judged RELATIVE to the validated baseline SMC on that plant, because the
  cold start itself can overshoot (diode pre-charge + LC ring: Vin 14 V, Vref 20 V gives
  1.35-1.5 x Vref for the baseline, whatever the controller does):
  finite, IL >= 0, Vmax <= max(1.2 Vref, 1.05 x baseline Vmax),
  IL_max <= max(11.5 A, 1.05 x baseline), settles (+-2 %) before max(30 ms, 1.5 x baseline + 2 ms)
  and stays there, duty peak-to-peak over the last 10 ms < 0.005 (no limit cycle).
Plus the analytic discrete current-loop gain  g = (K/phi) * dt * Vout / L  at
Vout = 40 V (a bad transient), which must stay < 1 (instability at 2).

    python experiments/analyze_smc_bounds.py            # candidate table
"""
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from controllers.smc import SlidingModeController
from converter import BoostConverter

BASE = np.array([config.SMC_KV, config.SMC_KI, config.SMC_GAIN, config.SMC_BOUNDARY])
PLANTS = [(12.0, 20.0, 24.0)] + list(itertools.product((10.0, 14.0), (10.0, 30.0), (20.0, 28.0)))


def simulate(params, vin, R, vref, t_end=0.06):
    Kv, Ki, K, phi = params
    conv = BoostConverter(Vin=vin, R=R, Vref=vref)
    conv.reset(IL=0.0, Vout=0.0)
    smc = SlidingModeController(Kv=Kv, Ki=Ki, switching_gain=K, boundary_layer=phi)
    n = int(round(t_end / conv.timestep))
    V, IL, D = np.zeros(n), np.zeros(n), np.zeros(n)
    for k in range(n):
        d = smc.update(conv.Vref, conv.Vout, conv.timestep, conv.IL)
        conv.step(d)
        V[k], IL[k], D[k] = conv.Vout, conv.IL, conv.duty
    t = (np.arange(n) + 1) * conv.timestep
    out = np.abs(V - vref) > 0.02 * vref
    settle = t[np.nonzero(out)[0][-1]] if out.any() else 0.0
    if out[-1]:
        settle = np.inf
    return dict(finite=bool(np.all(np.isfinite(V))), vmax=V.max() / vref, ilmax=IL.max(),
                ilmin=IL.min(), settle=settle, p2p=D[-200:].max() - D[-200:].min())


_BASE_CACHE = {}


def baseline(plant):
    if plant not in _BASE_CACHE:
        _BASE_CACHE[plant] = simulate(BASE, *plant)
    return _BASE_CACHE[plant]


def judge(r, plant=None):
    b = baseline(plant) if plant is not None else dict(vmax=0.0, ilmax=0.0, settle=0.0)
    return (r["finite"] and r["vmax"] <= max(1.2, 1.05 * b["vmax"])
            and r["ilmax"] <= max(11.5, 1.05 * b["ilmax"]) and r["ilmin"] >= -1e-9
            and r["settle"] <= max(0.030, 1.5 * b["settle"] + 0.002) and r["p2p"] < 0.005)


def current_loop_gain(K, phi, v=40.0):
    return (K / phi) * config.DT * v / config.L


def check_box(lo, hi, n=2, ratio_cap=None):
    """Worst-case summary over the box (n=2: corners, n=3: + mid-points).

    ratio_cap: optional cap on (K/phi) as a multiple of the baseline K/phi; combinations
    above it have phi raised until the cap holds (the same joint clamp as AdaptiveSMC).
    """
    axes = [np.exp(np.linspace(np.log(l), np.log(h), n)) for l, h in zip(lo, hi)]
    axes = [np.unique(np.append(a, 1.0)) if n == 3 else a for a in axes]
    worst = dict(settle=0.0, vmax=0.0, ilmax=0.0, p2p=0.0, gain=0.0, fails=0, total=0)
    for s in itertools.product(*axes):
        p = BASE * np.array(s)
        if ratio_cap is not None:
            p[3] = max(p[3], p[2] / (ratio_cap * BASE[2] / BASE[3]))
        worst["gain"] = max(worst["gain"], current_loop_gain(p[2], p[3]))
        for plant in PLANTS:
            r = simulate(p, *plant)
            worst["total"] += 1
            worst["fails"] += int(not judge(r, plant))
            worst["settle"] = max(worst["settle"], r["settle"])
            worst["vmax"] = max(worst["vmax"], r["vmax"])
            worst["ilmax"] = max(worst["ilmax"], r["ilmax"])
            worst["p2p"] = max(worst["p2p"], r["p2p"])
    return worst


if __name__ == "__main__":
    b = simulate(BASE, 12.0, 20.0, 24.0)
    print("baseline nominal cold start:", {k: round(float(v), 4) for k, v in b.items()})
    ws = [simulate(BASE, *pl) for pl in PLANTS]
    print("baseline on all 9 extreme plants: pass =", sum(judge(w, pl) for w, pl in zip(ws, PLANTS)), "/", len(ws),
          "| worst settle %.1f ms, worst IL %.1f A" % (1e3 * max(w["settle"] for w in ws),
                                                       max(w["ilmax"] for w in ws)))
    print("baseline current-loop gain at 40 V: %.2f (limit 2.0)\n" % current_loop_gain(
        config.SMC_GAIN, config.SMC_BOUNDARY))
    print(f"{'box (Kv,Ki +-)':>14} {'box (K,phi)':>16} {'fails':>9} {'worst settle ms':>16} "
          f"{'worst Vmax/Vref':>16} {'worst IL A':>10} {'worst p2p':>10} {'loop gain@40V':>14}")
    for ra, rb_lo, rb_hi in [(1.25, 0.9, 1.1), (1.4, 0.85, 1.15), (1.5, 0.85, 1.2),
                             (1.75, 0.8, 1.25), (2.0, 0.8, 1.25)]:
        lo = (1 / ra, 1 / ra, rb_lo, 1.0)
        hi = (ra, ra, rb_hi, 1.0 / rb_lo)         # phi box mirrors the K box inverted
        w = check_box(lo, hi, n=2)
        print(f"  Kv,Ki x[{1/ra:.2f},{ra:.2f}]  K x[{rb_lo:.2f},{rb_hi:.2f}] phi x[1.00,{1/rb_lo:.2f}]  "
              f"{w['fails']:>3}/{w['total']:<4} {1e3*w['settle']:>14.1f} {w['vmax']:>16.3f} "
              f"{w['ilmax']:>10.2f} {w['p2p']:>10.4f} {w['gain']:>14.2f}", flush=True)


def refine():
    print("\nStage 7 candidate boxes WITH a joint cap on K/phi (corners + mid-points, n=3):")
    for name, lo, hi, cap in [
            ("A", (0.8, 0.8, 0.9, 0.9), (1.25, 1.25, 1.1, 1.15), 1.2),
            ("B", (0.8, 0.8, 0.9, 0.85), (1.25, 1.25, 1.1, 1.25), 1.2),
            ("C", (0.85, 0.85, 0.9, 0.9), (1.2, 1.2, 1.1, 1.15), 1.2)]:
        w = check_box(lo, hi, n=3, ratio_cap=cap)
        print(f"  {name}: Kv,Ki x[{lo[0]},{hi[0]}] K x[{lo[2]},{hi[2]}] phi x[{lo[3]},{hi[3]}] cap {cap}:"
              f" fails {w['fails']}/{w['total']}  worst settle {1e3*w['settle']:.1f} ms  "
              f"worst p2p {w['p2p']:.4f}  worst IL {w['ilmax']:.2f} A  loop gain@40V {w['gain']:.2f}",
              flush=True)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "refine":
    refine()
