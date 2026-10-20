"""Safe gain-range analysis for the adaptive PID (Stage 6).

For scale factors (sKp, sKi, sKd) applied to the baseline PID, compute the
worst-case phase / gain margin over the 27-point operating grid used in Stage 2
(Vin 10/12/14 x R 10/20/30 x Vref 20/24/28, delay 1.5 Ts).  Reports the margins
at every corner of the box [1/r, r]^3 (the worst case is always at a corner
because the margins are monotone in each gain here - verified numerically).
"""
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from experiments.tune_pid import DELAY, PLANTS, controller_fr, margins


def worst_margins(Kp, Ki, Kd):
    Cw = controller_fr(Kp, Ki, Kd)
    pm, gm = 1e9, 1e9
    for G in PLANTS.values():
        wc, p, g = margins(Cw * G * DELAY)
        if np.isnan(wc):
            return -180.0, -99.0
        pm, gm = min(pm, p), min(gm, g)
    return pm, gm


def box_worst(rp, ri, rd, n=3):
    base = np.array([config.PID_KP, config.PID_KI, config.PID_KD])
    worst = (1e9, 1e9)
    for sp, si, sd in itertools.product((1 / rp, 1, rp), (1 / ri, 1, ri), (1 / rd, 1, rd)):
        pm, gm = worst_margins(*(base * [sp, si, sd]))
        worst = (min(worst[0], pm), min(worst[1], gm))
    return worst


if __name__ == "__main__":
    base = np.array([config.PID_KP, config.PID_KI, config.PID_KD])
    print("baseline worst PM/GM: %.1f deg / %.1f dB" % worst_margins(*base))
    print("\nsingle-gain sweeps (others at baseline): worst PM [deg] / GM [dB] over the grid")
    for name, idx in (("Kp", 0), ("Ki", 1), ("Kd", 2)):
        row = []
        for s in (0.25, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0):
            k = base.copy(); k[idx] *= s
            pm, gm = worst_margins(*k)
            row.append(f"x{s:<4g}: {pm:6.1f}/{gm:5.1f}")
        print(f"  {name}: " + " | ".join(row))
    print("\nbox [1/r, r] on all three gains simultaneously (worst corner):")
    for r in (1.25, 1.5, 2.0):
        pm, gm = box_worst(r, r, r)
        print(f"  r = {r:<4g}: worst PM {pm:6.1f} deg, worst GM {gm:5.1f} dB")


def box_check(lo, hi, n=5):
    """Worst PM/GM over an n^3 log-spaced grid inside the box [lo*base, hi*base]."""
    base = np.array([config.PID_KP, config.PID_KI, config.PID_KD])
    axes = [np.exp(np.linspace(np.log(l), np.log(h), n)) for l, h in zip(lo, hi)]
    # make sure the baseline (scale 1) itself is on every axis
    axes = [np.unique(np.append(a, 1.0)) for a in axes]
    pm, gm = 1e9, 1e9
    for s in itertools.product(*axes):
        p, g = worst_margins(*(base * np.array(s)))
        pm, gm = min(pm, p), min(gm, g)
    return pm, gm


def search_box(pm_min=20.0, gm_min=2.0):
    """Largest (log-volume) box from a small candidate set meeting the margins."""
    best = None
    for kp_lo, kp_hi, ki_lo, ki_hi, kd_lo, kd_hi in itertools.product(
            (0.25, 0.5), (1.5, 2.0), (0.25, 0.5), (1.0, 1.1, 1.2, 1.3),
            (0.8, 0.9, 1.0), (1.1, 1.25, 1.5)):
        lo, hi = (kp_lo, ki_lo, kd_lo), (kp_hi, ki_hi, kd_hi)
        # cheap reject: single-axis extremes first
        pm, gm = box_check(lo, hi, n=3)
        if pm < pm_min or gm < gm_min:
            continue
        vol = sum(np.log(h / l) for l, h in zip(lo, hi))
        if best is None or vol > best[0]:
            best = (vol, lo, hi)
    return best


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "search":
    vol, lo, hi = search_box()
    print("\nlargest box with worst PM >= 20 deg and GM >= 2 dB (3^3 screen):")
    print("  lo =", lo, " hi =", hi)
    print("  5^3 verification: worst PM %.1f deg, GM %.1f dB" % box_check(lo, hi, n=5))
    base = np.array([config.PID_KP, config.PID_KI, config.PID_KD])
    print("  gains: Kp [%.5g, %.5g]  Ki [%.5g, %.5g]  Kd [%.5g, %.5g]" % tuple(
        v for l, h, b in zip(lo, hi, base) for v in (l * b, h * b)))


def search_tight_box(pm_min=40.0, gm_min=4.0):
    """Stage 6b: conservative box. Largest log-volume candidate whose full 5^3 grid keeps
    worst-case PM >= pm_min and GM >= gm_min over the 27 operating points."""
    best = None
    cands = itertools.product((0.7, 0.85, 1.0), (1.15, 1.3, 1.5),      # Kp lo, hi
                              (0.7, 0.85, 1.0), (1.0, 1.05, 1.1),      # Ki lo, hi
                              (0.9, 0.95, 1.0), (1.05, 1.1, 1.2))      # Kd lo, hi
    for kp_lo, kp_hi, ki_lo, ki_hi, kd_lo, kd_hi in cands:
        lo, hi = (kp_lo, ki_lo, kd_lo), (kp_hi, ki_hi, kd_hi)
        pm, gm = box_check(lo, hi, n=3)
        if pm < pm_min or gm < gm_min:
            continue
        pm5, gm5 = box_check(lo, hi, n=5)
        if pm5 < pm_min or gm5 < gm_min:
            continue
        vol = sum(np.log(h / l) for l, h in zip(lo, hi))
        if best is None or vol > best[0]:
            best = (vol, lo, hi, pm5, gm5)
    return best


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "tight":
    print("margins of previous (Stage 6a) box lo=(0.25,0.25,0.8) hi=(2,1,1.5):",
          "PM %.1f deg, GM %.1f dB" % box_check((0.25, 0.25, 0.8), (2.0, 1.0, 1.5)))
    best = search_tight_box()
    vol, lo, hi, pm, gm = best
    base = np.array([config.PID_KP, config.PID_KI, config.PID_KD])
    print("\nStage 6b: largest box with worst PM >= 40 deg and GM >= 4 dB (5^3 verified)")
    print("  factors lo =", lo, " hi =", hi, "| worst PM %.1f deg, GM %.1f dB" % (pm, gm))
    print("  gains: Kp [%.5g, %.5g]  Ki [%.5g, %.5g]  Kd [%.5g, %.5g]" % tuple(
        v for l, h, b in zip(lo, hi, base) for v in (l * b, h * b)))
    for name, (l, h) in zip(("sweep r=1.25 box", "r=1.5 box", "r=2 box"), ((1/1.25, 1.25), (1/1.5, 1.5), (0.5, 2.0))):
        print(f"  (for reference) symmetric {name}: ", "PM %.1f deg, GM %.1f dB" % box_check((l,)*3, (h,)*3))
