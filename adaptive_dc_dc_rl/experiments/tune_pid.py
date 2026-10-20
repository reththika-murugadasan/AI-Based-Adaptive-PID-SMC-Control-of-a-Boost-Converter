"""PID tuning for the boost converter (Stage 2).

STEP 1 - plant.  Linearise the averaged CCM model about the operating point
(D, Vo = Vin/(1-D), IL = Vo/(R(1-D))):
    A = [[0, -D'/L], [D'/C, -1/(RC)]],   B = [Vo/L, -IL/C]^T,   C = [0 1]
giving Gvd(s) = vo/d with DC gain Vo/D', a lightly damped LC pole pair at
w0 = D'/sqrt(LC) and a RIGHT-HALF-PLANE zero at wz = R D'^2 / L.  The RHP zero
caps usable bandwidth (rule of thumb: crossover < wz/5), and the pole pair
(Q ~ 7) needs damping from the controller -> the derivative term is needed.

STEP 2 - frequency-domain design over a GRID of operating points
(Vin 10/12/14 V x R 10/20/30 ohm x Vref 20/24/28 V, D = 1 - Vin/Vref).
Loop L(s) = C_pid(s) * Gvd(s) * exp(-1.5 Ts s)   (ZOH + compute delay).
Maximise the nominal crossover frequency subject to
    min phase margin >= PM_MIN,  min gain margin >= GM_MIN_DB  over the grid,
and a limit on the controller's high-frequency gain.  (Differential evolution.)

STEP 3 - nonlinear time-domain check on the real simulator and a SHORT local
refinement of a time-domain cost (IAE + overshoot penalty over the five
required scenarios).  Results are printed; copy them into config.py.
"""
import itertools
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution, minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

PM_MIN, GM_MIN_DB = 45.0, 6.0
W = np.logspace(0.5, 4.6, 1500)           # rad/s


def plant_tf(Vin, R, Vref):
    """Return Gvd(jw) on the global grid W for the operating point."""
    D = 1.0 - Vin / Vref
    Dp = 1.0 - D
    Vo = Vin / Dp
    IL = Vo / (R * Dp)
    A = np.array([[0, -Dp / config.L], [Dp / config.C, -1 / (R * config.C)]])
    B = np.array([[Vo / config.L], [-IL / config.C]])
    Cm = np.array([[0.0, 1.0]])
    s = 1j * W
    G = np.array([(Cm @ np.linalg.solve(si * np.eye(2) - A, B))[0, 0] for si in s])
    return G


def controller_fr(Kp, Ki, Kd):
    s = 1j * W
    return Kp + Ki / s + Kd * s / (1 + config.PID_DERIV_TAU * s)


def margins(Lw):
    """Gain crossover, phase margin [deg], gain margin [dB] from L(jw) samples."""
    mag = np.abs(Lw)
    ph = np.unwrap(np.angle(Lw))
    idx = np.nonzero((mag[:-1] >= 1) & (mag[1:] < 1))[0]
    if idx.size == 0:
        return np.nan, -180.0, 0.0
    k = idx[0]                                # first gain crossover
    wc = W[k]
    pm = np.degrees(ph[k]) + 180.0
    pm = (pm + 180) % 360 - 180
    # gain margin at the first -180 deg crossing after wc
    ph_deg = np.degrees(ph)
    cr = np.nonzero((ph_deg[k:-1] + 180) * (ph_deg[k + 1:] + 180) <= 0)[0]
    gm = 60.0 if cr.size == 0 else -20 * np.log10(mag[k + cr[0]])
    return wc, pm, gm


GRID = [(vi, r, vr) for vi, r, vr in itertools.product((10, 12, 14), (10, 20, 30),
                                                       (20, 24, 28))]
PLANTS = {g: plant_tf(*g) for g in GRID}
DELAY = np.exp(-1.5 * config.T_SW * 1j * W)


WC_CAP = 600.0     # rad/s; RHP zero is at >= 2500 rad/s over the grid (wz/4)


def grid_margins(logk):
    Kp, Ki, Kd = np.exp(logk)
    Cw = controller_fr(Kp, Ki, Kd)
    pm_min, gm_min, wc_nom = 1e9, 1e9, 0.0
    for g, G in PLANTS.items():
        wc, pm, gm = margins(Cw * G * DELAY)
        if np.isnan(wc):
            return None
        pm_min, gm_min = min(pm_min, pm), min(gm_min, gm)
        if g == (12, 20, 24):
            wc_nom = wc
    return wc_nom, pm_min, gm_min


def margin_penalty(m):
    return 100.0 * (max(0, PM_MIN - m[1]) + max(0, GM_MIN_DB - m[2]))


def design_cost(logk):
    m = grid_margins(logk)
    if m is None:
        return 1e4
    wc_nom, pm_min, gm_min = m
    # fastest nominal loop up to WC_CAP, subject to margins; tie-break: more PM
    return -min(wc_nom, WC_CAP) / WC_CAP + margin_penalty(m) - 0.001 * min(pm_min, 70)


def simulate_scenario(gains, scen, t_dist=0.1, t_end=0.2):
    """Nonlinear closed loop. Returns t, vout, il, duty, vref_array."""
    from converter import BoostConverter
    from controllers.pid import PIDController
    kind, val = scen
    conv = BoostConverter()
    conv.reset(IL=0.0, Vout=0.0)
    pid = PIDController(*gains)
    n = int(round(t_end / conv.timestep)); kd = int(round(t_dist / conv.timestep))
    out = np.zeros((n, 4))
    for k in range(n):
        if k == kd:
            if kind == "vin": conv.set_input_voltage(val)
            elif kind == "load": conv.set_load(val)
        d = pid.update(conv.Vref, conv.Vout)
        conv.step(d)
        out[k] = (conv.time, conv.Vout, conv.IL, conv.duty)
    return out


SCENARIOS = [("nom", None), ("vin", 10.0), ("vin", 14.0), ("load", 10.0), ("load", 30.0)]


KP_MIN = 2e-3      # keep a genuine proportional term (avoid degenerate I+D)


def time_cost(logk):
    gains = np.exp(logk)
    m = grid_margins(logk)
    if m is None:
        return 1e4
    cost = margin_penalty(m) + 1e3 * max(0.0, np.log(KP_MIN) - logk[0])
    kd = int(round(0.1 / config.DT))
    try:
        for sc in SCENARIOS:
            o = simulate_scenario(gains, sc)
            v = o[:, 1]
            if not np.all(np.isfinite(v)) or v.max() > 80:
                return 1e4
            err = np.abs(config.VREF - v)
            cost += err.mean()                                        # IAE, whole run
            cost += 0.15 * max(0.0, v.max() - config.VREF)            # startup overshoot
            cost += 0.3 * err[kd:].max()                              # worst deviation after dist.
            cost += 4.0 * np.mean(err[kd:] > 0.02 * config.VREF)      # time outside +-2 % after dist.
            cost += 5.0 * err[-200:].mean()                           # steady-state
    except Exception:
        return 1e4
    return cost


def main():
    if len(sys.argv) == 4:                                  # start from given gains
        x0 = np.log([float(a) for a in sys.argv[1:4]])
        print("Skipping frequency-domain search; starting from", np.exp(x0))
    else:
        print("== Step 2: frequency-domain robust design ==")
        bounds = [(np.log(1e-3), np.log(1.0)), (np.log(0.1), np.log(500.0)),
                  (np.log(1e-7), np.log(1e-3))]
        res = differential_evolution(design_cost, bounds, seed=0, maxiter=80, popsize=15,
                                     tol=1e-6, polish=False)
        x0 = res.x
        Kp, Ki, Kd = np.exp(x0)
        wc, pm, gm = grid_margins(x0)
        print(f"  design gains: Kp={Kp:.5g} Ki={Ki:.5g} Kd={Kd:.5g}")
        print(f"  nominal wc = {wc:.0f} rad/s; worst PM over grid = {pm:.1f} deg,"
              f" worst GM = {gm:.1f} dB", flush=True)
    print("== Step 3: nonlinear time-domain refinement (Nelder-Mead) ==")
    print(f"  time-domain cost at start: {time_cost(x0):.3f}", flush=True)
    r2 = minimize(time_cost, x0, method="Nelder-Mead",
                  options=dict(xatol=1e-3, fatol=1e-4, maxiter=150))
    K2 = np.exp(r2.x)
    wc, pm, gm = grid_margins(r2.x)
    print(f"  refined gains: Kp={K2[0]:.5g} Ki={K2[1]:.5g} Kd={K2[2]:.5g}  cost={r2.fun:.3f}")
    print(f"  refined: nominal wc = {wc:.0f} rad/s, worst PM = {pm:.1f} deg, worst GM = {gm:.1f} dB")


if __name__ == "__main__":
    main()
