"""Baseline-recovery supervisor shared by adaptive controllers.

Decides when the RL agent must be overridden and the controller parameters driven
back to the validated baseline.  Triggers:
  * persistent tracking error: low-pass filtered |e| > err_thr for ``persist`` seconds
  * unsafe state (immediate): inductor current > il_warn or Vout > v_warn_ratio * Vref
  * chatter (optional): low-pass filtered |dD| per step > chatter_thr for ``chatter_persist``
Release: filtered |e| < release_thr for ``release_hold`` seconds, after ``min_hold``.
(The adaptive PID has the same logic built in; it is left untouched as validated code.)
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class SupervisorConfig:
    err_thr: float
    err_tau: float
    persist: float
    il_warn: float
    v_warn_ratio: float
    min_hold: float
    release_thr: float
    release_hold: float
    chatter_thr: float = math.inf
    chatter_tau: float = 1e-3
    chatter_persist: float = 10e-3


class RecoverySupervisor:
    def __init__(self, cfg: SupervisorConfig):
        self.cfg = cfg
        self.reset()

    def reset(self):
        self.recovering = False
        self.events = 0
        self.reason = ""
        self._err_f = 0.0
        self._above = 0.0
        self._chat_f = 0.0
        self._chat_t = 0.0
        self._hold = 0.0
        self._calm = 0.0
        self._prev_duty = None

    def step(self, error, vout, reference, inductor_current, duty, dt):
        c = self.cfg
        self._err_f += (abs(error) - self._err_f) * dt / max(c.err_tau, dt)
        if duty is not None:                     # chatter filter needs two consecutive duties
            if self._prev_duty is not None:
                self._chat_f += (abs(duty - self._prev_duty) - self._chat_f) * dt / max(c.chatter_tau, dt)
            self._prev_duty = duty
        self._chat_t = self._chat_t + dt if self._chat_f > c.chatter_thr else 0.0
        unsafe = (inductor_current is not None and inductor_current > c.il_warn) \
            or vout > c.v_warn_ratio * reference
        if not self.recovering:
            self._above = self._above + dt if self._err_f > c.err_thr else 0.0
            why = "unsafe" if unsafe else "error" if self._above >= c.persist else \
                "chatter" if self._chat_t >= c.chatter_persist else ""
            if why:
                self.recovering, self.reason = True, why
                self.events += 1
                self._hold, self._calm = c.min_hold, 0.0
        else:
            self._hold = max(0.0, self._hold - dt)
            calm = self._err_f < c.release_thr and not unsafe and self._chat_f <= c.chatter_thr
            self._calm = self._calm + dt if calm else 0.0
            if self._hold == 0.0 and self._calm >= c.release_hold:
                self.recovering, self.reason = False, ""
                self._above = self._chat_t = 0.0
        return self.recovering
