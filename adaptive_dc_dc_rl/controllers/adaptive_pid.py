"""Adaptive PID: an RL agent tunes the gains of the (unchanged) baseline PID.

    action a in [-1, 1]^3 -> requested (Kp, Ki, Kd) -> clamp to the safe box
        -> smoothing / rate limit -> PIDController.set_gains()
    PIDController (anti-windup, derivative filter, duty limits) -> duty

Action -> gain mapping (per gain, K0 = baseline, [lo, hi] = safe range):
    a >= 0 :  K = K0 * (hi / K0) ** a
    a <  0 :  K = K0 * (lo / K0) ** (-a)
so a = 0 is EXACTLY the validated PID and +/-1 reach the range ends.  The mapping
is relative/logarithmic because Kp, Ki, Kd differ by six orders of magnitude and the
stability margin itself depends multiplicatively on them.

Safety layers
1. Safe box (config.ADAPT_RANGE_FACTORS x baseline).  ``clamp_gains`` is applied to
   EVERY proposed gain set, whether it comes from the action mapping or from
   ``propose_gains``.  The box must lie inside the CERTIFIED box
   (config.ADAPT_CERTIFIED_FACTORS, from the margin analysis) unless
   ``allow_uncertified=True`` - widening it by accident raises an error.
2. Smoothing + rate limit on ln K once per ``adapt`` call:
       ln K_new = ln K_prev + clip(alpha (ln K_target - ln K_prev), +/-rate_limit)
3. Baseline recovery (optional).  A supervisor, run at every PID step, engages when
   (a) the low-pass filtered |error| stays above ERR_THR for PERSIST seconds, or
   (b) IL > IL_WARN or Vout > V_WARN_RATIO * Vref (immediately).
   While engaged the agent's request is ignored and the target is the baseline gains;
   the gains return through the same rate limit (smooth, never a jump).  It releases
   after the filtered error stays below RELEASE_THR for RELEASE_HOLD (and MIN_HOLD).
4. The PID keeps its own anti-windup and [0.05, 0.95] duty saturation; its integral
   state is stored in duty units so a gain change never makes the duty jump.
"""
from __future__ import annotations

import math

import numpy as np

import config
from controllers.pid import PIDController

GAINS = ("Kp", "Ki", "Kd")


def default_ranges():
    base = (config.PID_KP, config.PID_KI, config.PID_KD)
    return tuple(tuple(b * f for f in config.ADAPT_RANGE_FACTORS[g]) for b, g in zip(base, GAINS))


class AdaptivePID:
    def __init__(self, base=None, ranges=None, alpha=config.ADAPT_SMOOTH_ALPHA,
                 rate_limit=config.ADAPT_RATE_LIMIT, pid=None, recovery=None,
                 allow_uncertified=False, **pid_kwargs):
        self.base = dict(zip(GAINS, base if base is not None else
                             (config.PID_KP, config.PID_KI, config.PID_KD)))
        rng = ranges if ranges is not None else default_ranges()
        self.ranges = dict(zip(GAINS, (tuple(r) for r in rng)))
        for g in GAINS:
            lo, hi = self.ranges[g]
            if not (0 < lo <= self.base[g] <= hi):
                raise ValueError(f"{g}: need 0 < lo <= baseline <= hi, got {lo}, {self.base[g]}, {hi}")
            if not allow_uncertified:
                clo, chi = (self.base[g] * f for f in config.ADAPT_CERTIFIED_FACTORS[g])
                if lo < clo * (1 - 1e-9) or hi > chi * (1 + 1e-9):
                    raise ValueError(f"{g} range ({lo:.5g}, {hi:.5g}) exceeds the certified safe box "
                                     f"({clo:.5g}, {chi:.5g}); pass allow_uncertified=True to override")
        if not (0 < alpha <= 1) or rate_limit <= 0:
            raise ValueError("alpha must be in (0, 1] and rate_limit > 0")
        self.alpha, self.rate_limit = float(alpha), float(rate_limit)
        self.recovery_enabled = config.ADAPT_RECOVERY_ENABLED if recovery is None else bool(recovery)
        self.pid = pid or PIDController(*self.base.values(), **pid_kwargs)
        self.reset()

    # ----------------------------------------------------------------- state
    def reset(self):
        self.pid.reset()
        self.gains = dict(self.base)
        self.pid.set_gains(**self.gains)
        self.requested = dict(self.base)
        self.recovering = False
        self.recovery_events = 0
        self._err_f = 0.0                 # low-pass filtered |error|
        self._above = 0.0                 # time the filtered error has been above the threshold
        self._hold = 0.0                  # remaining minimum engage time
        self._calm = 0.0                  # time the filtered error has been below the release level

    @property
    def gain_array(self):
        return np.array([self.gains[g] for g in GAINS])

    # ------------------------------------------------------------ safe box
    def clamp_gains(self, gains):
        """Clamp a proposed gain set to the safe box (non-finite values -> baseline)."""
        out = {}
        for g in GAINS:
            k = float(gains[g]) if g in gains else self.base[g]
            if not math.isfinite(k):
                k = self.base[g]
            lo, hi = self.ranges[g]
            out[g] = min(max(k, lo), hi)
        return out

    def target_gains(self, action):
        """Normalised action (3,) -> requested gains, clamped to the safe box."""
        a = np.clip(np.nan_to_num(np.asarray(action, float).reshape(-1), nan=0.0,
                                  posinf=1.0, neginf=-1.0), -1.0, 1.0)
        if a.size != 3:
            raise ValueError("adaptive-PID action must have 3 elements")
        out = {}
        for ai, g in zip(a, GAINS):
            lo, hi = self.ranges[g]
            k0 = self.base[g]
            out[g] = k0 * (hi / k0) ** ai if ai >= 0 else k0 * (lo / k0) ** (-ai)
        return self.clamp_gains(out)

    def normalized_state(self):
        """Current gains as the action that would request them (in [-1, 1])."""
        s = []
        for g in GAINS:
            lo, hi = self.ranges[g]
            k0, k = self.base[g], self.gains[g]
            if k >= k0:
                s.append(0.0 if hi == k0 else math.log(k / k0) / math.log(hi / k0))
            else:
                s.append(math.log(k / k0) / math.log(k0 / lo))
        return np.clip(s, -1.0, 1.0)

    # ----------------------------------------------------------- adaptation
    def _move_toward(self, requested):
        for g in GAINS:
            lo, hi = self.ranges[g]
            cur = self.gains[g]
            step = self.alpha * (math.log(requested[g]) - math.log(cur))
            step = min(max(step, -self.rate_limit), self.rate_limit)
            self.gains[g] = min(max(cur * math.exp(step), lo), hi)
        self.pid.set_gains(**self.gains)
        return dict(self.gains)

    def adapt(self, action):
        """Move the gains toward the action's request (or toward baseline when recovering)."""
        self.requested = dict(self.base) if self.recovering else self.target_gains(action)
        return self._move_toward(self.requested)

    def propose_gains(self, Kp=None, Ki=None, Kd=None):
        """Apply an explicit gain proposal: clamp to the safe box, then smooth/rate-limit."""
        prop = {g: v for g, v in zip(GAINS, (Kp, Ki, Kd)) if v is not None}
        self.requested = dict(self.base) if self.recovering else self.clamp_gains(
            {**self.gains, **prop})
        return self._move_toward(self.requested)

    # ------------------------------------------------------------- recovery
    def _supervise(self, error, vout, reference, inductor_current, dt):
        tau = max(config.ADAPT_REC_ERR_TAU, dt)
        self._err_f += (abs(error) - self._err_f) * dt / tau
        unsafe = (inductor_current is not None and inductor_current > config.ADAPT_REC_IL_WARN) \
            or vout > config.ADAPT_REC_V_WARN_RATIO * reference
        if not self.recovering:
            self._above = self._above + dt if self._err_f > config.ADAPT_REC_ERR_THR else 0.0
            if unsafe or self._above >= config.ADAPT_REC_PERSIST:
                self.recovering = True
                self.recovery_events += 1
                self._hold, self._calm = config.ADAPT_REC_MIN_HOLD, 0.0
        else:
            self._hold = max(0.0, self._hold - dt)
            calm = self._err_f < config.ADAPT_REC_RELEASE_THR and not unsafe
            self._calm = self._calm + dt if calm else 0.0
            if self._hold == 0.0 and self._calm >= config.ADAPT_REC_RELEASE_HOLD:
                self.recovering = False
                self._above = 0.0

    def update(self, reference, measurement, dt=None, inductor_current=None):
        """PID step with the current gains; returns the duty (already saturated)."""
        dt_eff = self.pid.dt if dt is None else float(dt)
        if self.recovery_enabled and math.isfinite(reference) and math.isfinite(measurement):
            self._supervise(reference - measurement, measurement, reference, inductor_current, dt_eff)
        return self.pid.update(reference, measurement, dt)
