"""Stage 3 sanity check: why the generic voltage surface s = de/dt + lambda*e fails.

With s = e_dot + lam*e = 0 and  e_dot = -((1-D) IL - V/R)/C  the duty is fixed
algebraically:   (1-D) = (V/R + C*lam*e) / IL .
This controls Vout directly; what remains is the inductor-current dynamics
(non-minimum-phase zero dynamics), which are unstable.  We simulate it.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from converter import BoostConverter


def run(lam=300.0, t_end=0.1):
    conv = BoostConverter()
    conv.reset(IL=2.4, Vout=24.0)          # start AT the equilibrium of D = 0.5
    conv.set_reference(26.0)               # small 2 V reference step
    V, IL, D = [], [], []
    for _ in range(int(t_end / conv.timestep)):
        e = conv.Vref - conv.Vout
        u = (conv.Vout / config.R + config.C * lam * e) / max(conv.IL, 0.05)
        conv.step(1.0 - u)
        V.append(conv.Vout); IL.append(conv.IL); D.append(conv.duty)
    return np.array(V), np.array(IL), np.array(D)


if __name__ == "__main__":
    V, IL, D = run()
    ok = abs(V[-1] - 26.0) < 1.0 and 0.5 < IL[-1] < 8.0
    print(f"naive s = e_dot + lam*e : final Vout = {V[-1]:.2f} V (ref 26), final IL = {IL[-1]:.2f} A, "
          f"duty range [{D.min():.2f}, {D.max():.2f}]")
    print("Vout tracked the reference with a sane IL" if ok else
          "FAILED: unstable zero dynamics (voltage collapse / current runaway)")
