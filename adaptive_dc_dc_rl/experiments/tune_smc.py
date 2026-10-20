"""SMC tuning (Stage 3).

Analytic part (fixed by derivation, see controllers/smc.py):
* Outer loop on the current-reference level.  With IL = Iref the plant is
      dv/dt = Vin/(C V0) * i - 2/(R C) * v     (gain 1064 V/(A s), pole 213 rad/s)
  PI with zero at the plant pole (Ki/Kv = 2/(RC) = 213 /s) gives a crossover
  w_c = 1064 Kv; w_c = 800 rad/s -> Kv = 0.75 A/V, Ki = 160 A/(V s).
* K = 0.30 covers |Vin - 12| / Vout (<= 0.1-0.17) with margin.
* phi = 0.8 A keeps the discrete current-loop gain (K/phi)(dt Vout/L)
  = 0.45 at 24 V (< 1) and < 1 up to ~53 V.

Numerical part: Nelder-Mead on (Kv, Ki) with the SAME time-domain cost terms
that were used for the PID (IAE, startup overshoot, worst deviation after the
disturbance, time outside +-2 %, final error) over the five scenarios.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from controllers.smc import SlidingModeController
from experiments.run_pid import run_scenario

SCENARIOS = [(None, None), ("vin", 10.0), ("vin", 14.0), ("load", 10.0), ("load", 30.0)]


def cost(logk):
    Kv, Ki = np.exp(logk)
    total = 0.0
    kd = int(round(0.1 / config.DT))
    for kind, val in SCENARIOS:
        r = run_scenario(kind, val, controller=SlidingModeController(Kv=Kv, Ki=Ki),
                         t_dist=0.1, t_end=0.2)
        v = r["vout"]
        if not np.all(np.isfinite(v)) or v.max() > 80:
            return 1e4
        err = np.abs(config.VREF - v)
        total += err.mean()
        total += 0.15 * max(0.0, v.max() - config.VREF)
        total += 0.3 * err[kd:].max()
        total += 4.0 * np.mean(err[kd:] > 0.02 * config.VREF)
        total += 5.0 * err[-200:].mean()
    return total


if __name__ == "__main__":
    x0 = np.log([config.SMC_KV, config.SMC_KI])
    print(f"analytic start Kv={config.SMC_KV} Ki={config.SMC_KI}  cost={cost(x0):.3f}",
          flush=True)
    r = minimize(cost, x0, method="Nelder-Mead", options=dict(xatol=1e-3, fatol=1e-4, maxiter=80))
    print(f"refined  Kv={np.exp(r.x[0]):.5g} Ki={np.exp(r.x[1]):.5g}  cost={r.fun:.3f}")
