"""Sliding-mode controller for the averaged boost converter.

Derivation (checked against converter/boost_converter.py, u = 1 - D):
    dIL/dt   = (Vin - u Vout) / L
    dVout/dt = (u IL - Vout / R) / C

1. A surface on the voltage alone (s = de/dt + lambda e, e = Vref - Vout) does
   not work: dVout/dt contains D directly, d(dVout/dt)/dD = -IL/C < 0, i.e.
   Vout has relative degree 1 and its instantaneous response to D has the
   OPPOSITE sign of the steady-state gain (right-half-plane zero).  Forcing
   such an s to zero puts the unstable zero dynamics (IL) inside the sliding
   manifold.  tests/test_smc.py confirms the sign on our model and
   experiments/smc_naive_check.py simulates the voltage-surface law (it fails).
2. The inductor current is minimum-phase w.r.t. D: d(dIL/dt)/dD = +Vout/L.
   With IL held at Iref the remaining Vout dynamics are
       C dVout/dt = Vin Iref / Vout - Vout / R,
   which is stable.  So the sliding surface is on the current:
       s = IL - Iref,   Iref = clip(Kv e + Ki xi, 0, I_max),   dxi/dt = e,
   i.e. the voltage error e = Vref - Vout enters through Iref.  On s = 0 the
   outer dynamics are the first-order plant  dv/dt = (Vin/(C V0)) i - (2/(RC)) v
   with PI feedback; Kv, Ki place its pole/zero (see tune_smc.py).
3. Control law.  ds/dt = (Vin - (1-D) Vout)/L - dIref/dt.  Equivalent control
   (ds/dt = 0, nominal Vin, dIref/dt neglected):  D_eq = 1 - Vin_nom / Vout.
   Reaching law: s > 0 (current too high) must lower IL, which needs a smaller
   D, hence
       D = D_eq - K sat(s / phi),   clipped to [duty_min, duty_max].
   Reaching condition s*ds/dt < 0 holds when K Vout / L exceeds the matched
   uncertainty |Vin - Vin_nom| / L + |dIref/dt| (K = 0.3 covers Vin 10-14 V
   against nominal 12 V at Vout >= 20 V; larger dIref/dt during transients is
   absorbed by the saturated control, leaving the surface only temporarily).
4. sat(s/phi) instead of sign(s) removes chattering.  Inside the boundary layer
   the law is a proportional current loop of gain K/phi [duty/A]; with the
   discrete step dt the current loop gain is (K/phi)(dt Vout / L), which must
   stay < 2 (stability) and preferably < 1: 0.375 * 1.2 = 0.45 at 24 V.

Reference-current limits:
* Iref >= 0 : the diode blocks reverse current, a negative request would only
  drive D to its minimum (never asked for).
* Iref <= I_max : limits startup inrush.
Iref is clipped, and the integral uses back-calculation anti-windup.
"""
from __future__ import annotations

import math

import config


def sat(x: float) -> float:
    """Saturation function: -1 for x<-1, x inside the layer, +1 for x>1."""
    return max(-1.0, min(1.0, x))


class SlidingModeController:
    uses_current = True              # tells the experiment runner to pass IL

    def __init__(self, Kv=config.SMC_KV, Ki=config.SMC_KI,
                 switching_gain=config.SMC_GAIN, boundary_layer=config.SMC_BOUNDARY,
                 i_max=config.SMC_I_MAX, vin_nom=config.SMC_VIN_NOM,
                 duty_min=config.DUTY_MIN, duty_max=config.DUTY_MAX,
                 aw_gain=config.SMC_AW_GAIN, dt=config.DT):
        if boundary_layer <= 0:
            raise ValueError("boundary_layer must be positive")
        self.Kv, self.Ki = float(Kv), float(Ki)
        self.switching_gain = float(switching_gain)
        self.boundary_layer = float(boundary_layer)
        self.i_max, self.vin_nom = float(i_max), float(vin_nom)
        self.duty_min, self.duty_max = float(duty_min), float(duty_max)
        self.aw_gain, self.dt = float(aw_gain), float(dt)
        self.reset()

    def reset(self):
        self.xi = 0.0                # integral state, in AMPS (= Ki * int e dt)
        self.last = dict(error=0.0, i_ref=0.0, s=0.0, d_eq=0.0, duty=0.0)

    # ------------------------------------------------------------------
    def current_reference(self, error):
        """(clipped Iref, unclipped Iref) from the voltage error and integral."""
        raw = self.Kv * error + self.xi
        return min(max(raw, 0.0), self.i_max), raw

    def sliding_surface(self, error, inductor_current):
        """s = IL - Iref(e)."""
        return inductor_current - self.current_reference(error)[0]

    def equivalent_duty(self, vout):
        """D_eq = 1 - Vin_nom / Vout (guarded against Vout -> 0)."""
        return 1.0 - self.vin_nom / max(vout, 0.5 * self.vin_nom)

    def update(self, reference, measurement, dt=None, inductor_current=0.0):
        """One control step; returns the duty cycle."""
        dt = self.dt if dt is None else float(dt)
        if not all(math.isfinite(x) for x in (reference, measurement, inductor_current)):
            raise ValueError("SMC received a non-finite input")
        e = reference - measurement
        i_ref, i_raw = self.current_reference(e)
        s = inductor_current - i_ref
        d_eq = self.equivalent_duty(measurement)
        duty = d_eq - self.switching_gain * sat(s / self.boundary_layer)
        duty = min(max(duty, self.duty_min), self.duty_max)

        # integral of the voltage error with back-calculation anti-windup
        self.xi += (self.Ki * e + self.aw_gain * (i_ref - i_raw)) * dt
        self.xi = min(max(self.xi, -self.i_max), 2.0 * self.i_max)
        self.last = dict(error=e, i_ref=i_ref, s=s, d_eq=d_eq, duty=duty)
        return duty
