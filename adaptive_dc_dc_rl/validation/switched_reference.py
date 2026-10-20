"""Switch-level ground truth: ideal switch + ideal diode, event-accurate.

Used ONLY for validating the averaged model.  Per switching period (D constant):
  on  (0..D*Ts)   : switch closed, diode blocked.  diL=Vin/L,  dv=-v/(RC)
  off (D*Ts..Ts)  : diode conducts while iL>0:  diL=(Vin-v)/L, dv=(iL-v/R)/C
                    when iL hits 0 (event) diode blocks: iL=0, dv=-v/(RC)
Returns the cycle-AVERAGED iL and v of every period for comparison.
"""
import numpy as np
from scipy.integrate import solve_ivp


def simulate_switched(duty, t_end, Vin, L, C, R, Ts, iL0=0.0, v0=None):
    v = Vin if v0 is None else v0
    iL = iL0
    n = int(round(t_end / Ts))
    t_c = (np.arange(n) + 0.5) * Ts
    iL_avg, v_avg = np.zeros(n), np.zeros(n)
    kw = dict(method="DOP853", rtol=1e-9, atol=1e-11)

    def seg(f, y0, T, events=None):
        return solve_ivp(f, (0, T), y0, events=events, dense_output=True, **kw)

    def hit(t, y):
        return y[0]
    hit.terminal, hit.direction = True, -1

    for k in range(n):
        ton, toff = duty * Ts, (1 - duty) * Ts
        s1 = seg(lambda t, y: [Vin / L, -y[1] / (R * C)], [iL, v], ton)
        parts = [(0.0, ton, s1)]
        iL1, v1 = s1.y[:, -1]
        if iL1 > 0 or v1 < Vin:
            s2 = seg(lambda t, y: [(Vin - y[1]) / L, (y[0] - y[1] / R) / C],
                     [iL1, v1], toff, events=hit)
            t2 = s2.t[-1]
            parts.append((ton, ton + t2, s2))
            iL2, v2 = s2.y[:, -1]
            if t2 < toff - 1e-15:                      # diode blocked: idle interval
                s3 = seg(lambda t, y: [0.0, -y[1] / (R * C)], [0.0, v2], toff - t2)
                parts.append((ton + t2, Ts, s3))
                iL2, v2 = 0.0, s3.y[1, -1]
        else:
            s3 = seg(lambda t, y: [0.0, -y[1] / (R * C)], [0.0, v1], toff)
            parts.append((ton, Ts, s3)); iL2, v2 = 0.0, s3.y[1, -1]
        ii = vv = 0.0
        for a, b, s in parts:                          # trapezoid cycle average
            tt = np.linspace(0, b - a, 40)
            y = s.sol(tt)
            ii += np.trapezoid(y[0], tt); vv += np.trapezoid(y[1], tt)
        iL_avg[k], v_avg[k] = ii / Ts, vv / Ts
        iL, v = max(iL2, 0.0), v2
    return t_c, iL_avg, v_avg
