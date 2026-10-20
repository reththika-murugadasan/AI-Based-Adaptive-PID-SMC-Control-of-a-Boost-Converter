"""Discrete PID voltage controller for the boost converter.

    e   = Vref - Vout
    u   = bias + Kp*e + I + Kd*d_filt
    I  += (Ki*e + Kb*(u_sat - u)) * dt        # back-calculation anti-windup
    duty = clip(u, out_min, out_max)

Design choices
* The integral state ``I`` is stored in DUTY units (= Ki * integral(e)), not as
  a raw integral of error.  Changing Ki at run time (needed by the adaptive RL
  controller later) therefore does not make the output jump.
* ``bias`` is a constant offset, default 0 (textbook PID).  A non-zero bias
  such as 0.5 was tried: it applies D = 0.5 at t = 0 while Vout = 0, an
  open-loop step into a lightly damped plant (43 % startup overshoot vs 2 %).
  It is never a Vin-dependent feed-forward.
* Derivative acts on the (filtered) measurement by default, so reference steps
  do not cause a derivative kick.  For a constant reference this is identical
  to de/dt.  First-order low-pass filter with time constant ``deriv_tau``.
* Anti-windup: back-calculation (rate 1/Tt = ``aw_gain``) plus a hard clamp of
  the integral state to the range that can still influence the output.
"""
from __future__ import annotations

import math

import config


class PIDController:
    def __init__(self, Kp=config.PID_KP, Ki=config.PID_KI, Kd=config.PID_KD,
                 dt=config.DT, bias=config.PID_BIAS,
                 out_min=config.DUTY_MIN, out_max=config.DUTY_MAX,
                 deriv_tau=config.PID_DERIV_TAU, aw_gain=config.PID_AW_GAIN,
                 derivative_on_measurement=True):
        self.Kp, self.Ki, self.Kd = float(Kp), float(Ki), float(Kd)
        self.dt, self.bias = float(dt), float(bias)
        self.out_min, self.out_max = float(out_min), float(out_max)
        self.deriv_tau, self.aw_gain = float(deriv_tau), float(aw_gain)
        self.derivative_on_measurement = derivative_on_measurement
        self.reset()

    # ---------------------------------------------------------------- API
    def reset(self, integral_state=0.0):
        self.i_term = float(integral_state)     # duty units
        self.d_filt = 0.0                       # filtered derivative [V/s]
        self._prev = None                       # previous measurement or error
        self.last = dict(error=0.0, p=0.0, i=0.0, d=0.0, unsat=self.bias,
                         duty=self.bias, saturated=False)

    def set_gains(self, Kp=None, Ki=None, Kd=None):
        if Kp is not None: self.Kp = float(Kp)
        if Ki is not None: self.Ki = float(Ki)
        if Kd is not None: self.Kd = float(Kd)

    def update(self, reference, measurement, dt=None):
        """One control step. Returns the duty cycle (already saturated)."""
        dt = self.dt if dt is None else float(dt)
        if not (math.isfinite(reference) and math.isfinite(measurement)):
            raise ValueError("PID received a non-finite reference/measurement")
        e = reference - measurement

        signal = -measurement if self.derivative_on_measurement else e
        if self._prev is None:
            raw = 0.0                           # no derivative on first sample
        else:
            raw = (signal - self._prev) / dt
        self._prev = signal
        a = self.deriv_tau / (self.deriv_tau + dt)
        self.d_filt = a * self.d_filt + (1.0 - a) * raw

        p = self.Kp * e
        d = self.Kd * self.d_filt
        u = self.bias + p + self.i_term + d
        duty = min(max(u, self.out_min), self.out_max)

        # integral update with back-calculation, then hard clamp
        self.i_term += (self.Ki * e + self.aw_gain * (duty - u)) * dt
        lo, hi = self.out_min - self.bias, self.out_max - self.bias
        self.i_term = min(max(self.i_term, lo), hi)

        self.last = dict(error=e, p=p, i=self.i_term, d=d, unsat=u, duty=duty,
                         saturated=(duty != u))
        return duty
