"""Adaptive SMC: an RL agent tunes four parameters of the (unchanged) sliding-mode law.

    action a in [-1,1]^4 -> requested (Kv, Ki, K, phi) -> clamp to the safe box
        -> smoothing / rate limit -> SlidingModeController parameters
    SlidingModeController (surface s = IL - Iref, D = D_eq - K sat(s/phi), anti-windup,
    current-reference and duty limits) -> duty

Adapted parameters (they directly shape tracking and switching; the law is untouched):
    Kv   current reference per volt of error    [A/V]   outer-loop proportional action
    Ki   current reference per volt-second      [A/V s] outer-loop integral action
    K    switching gain (duty units)                    reaching strength / disturbance bound
    phi  boundary-layer thickness [A]                   chattering vs. stiffness trade-off
Not adapted: I_max, Vin_nom, duty limits, anti-windup (safety constants).

Mapping per parameter (p0 = baseline, [lo, hi] = active range):
    a >= 0 : p = p0 (hi/p0)^a      a < 0 : p = p0 (lo/p0)^(-a)      (a = 0 -> exactly baseline)

Safety layers
1. Box clamp on every proposal, inside the CERTIFIED box (config.ASMC_CERTIFIED_FACTORS,
   verified by closed-loop simulation); wider ranges raise an error unless
   ``allow_uncertified=True``.
2. Joint constraint K/phi <= ASMC_MAX_RATIO_FACTOR x baseline K/phi: the discrete current
   loop has gain (K/phi)(dt Vout/L) per step and goes unstable at 2 (baseline 0.75 at
   40 V).  If violated, phi is raised (or K lowered if phi is at its top).  Applied to the
   targets AND to every smoothed step.
3. Smoothing + rate limit of ln p (alpha, max step per RL step).
4. RecoverySupervisor (controllers/safety_supervisor.py): persistent error, unsafe state or
   duty chatter -> agent ignored, parameters return to baseline through the rate limit.
"""
from __future__ import annotations

import math

import numpy as np

import config
from controllers.safety_supervisor import RecoverySupervisor, SupervisorConfig
from controllers.smc import SlidingModeController

PARAMS = ("Kv", "Ki", "K", "phi")
_ATTR = dict(Kv="Kv", Ki="Ki", K="switching_gain", phi="boundary_layer")


def default_ranges():
    base = _baseline()
    return {p: tuple(base[p] * f for f in config.ASMC_RANGE_FACTORS[p]) for p in PARAMS}


def _baseline():
    return dict(Kv=config.SMC_KV, Ki=config.SMC_KI, K=config.SMC_GAIN, phi=config.SMC_BOUNDARY)


def supervisor_config():
    return SupervisorConfig(
        err_thr=config.ASMC_REC_ERR_THR, err_tau=config.ASMC_REC_ERR_TAU,
        persist=config.ASMC_REC_PERSIST, il_warn=config.ASMC_REC_IL_WARN,
        v_warn_ratio=config.ASMC_REC_V_WARN_RATIO, min_hold=config.ASMC_REC_MIN_HOLD,
        release_thr=config.ASMC_REC_RELEASE_THR, release_hold=config.ASMC_REC_RELEASE_HOLD,
        chatter_thr=config.ASMC_REC_CHATTER_THR, chatter_tau=config.ASMC_REC_CHATTER_TAU,
        chatter_persist=config.ASMC_REC_CHATTER_PERSIST)


class AdaptiveSMC:
    uses_current = True

    def __init__(self, ranges=None, alpha=config.ASMC_SMOOTH_ALPHA,
                 rate_limit=config.ASMC_RATE_LIMIT, recovery=None, smc=None,
                 max_ratio_factor=config.ASMC_MAX_RATIO_FACTOR, allow_uncertified=False):
        self.base = _baseline()
        self.ranges = {p: tuple(r) for p, r in (ranges or default_ranges()).items()}
        for p in PARAMS:
            lo, hi = self.ranges[p]
            if not (0 < lo <= self.base[p] <= hi):
                raise ValueError(f"{p}: need 0 < lo <= baseline <= hi, got {lo}, {self.base[p]}, {hi}")
            if not allow_uncertified:
                clo, chi = (self.base[p] * f for f in config.ASMC_CERTIFIED_FACTORS[p])
                if lo < clo * (1 - 1e-9) or hi > chi * (1 + 1e-9):
                    raise ValueError(f"{p} range ({lo:.5g}, {hi:.5g}) exceeds the certified safe box "
                                     f"({clo:.5g}, {chi:.5g}); pass allow_uncertified=True to override")
        if not allow_uncertified and max_ratio_factor > config.ASMC_MAX_RATIO_FACTOR * (1 + 1e-9):
            raise ValueError("max_ratio_factor exceeds the certified K/phi cap")
        if not (0 < alpha <= 1) or rate_limit <= 0:
            raise ValueError("alpha must be in (0, 1] and rate_limit > 0")
        self.alpha, self.rate_limit = float(alpha), float(rate_limit)
        self.ratio_max = max_ratio_factor * self.base["K"] / self.base["phi"]
        self.recovery_enabled = config.ASMC_RECOVERY_ENABLED if recovery is None else bool(recovery)
        self.smc = smc or SlidingModeController()
        self.supervisor = RecoverySupervisor(supervisor_config())
        self.reset()

    # ----------------------------------------------------------------- state
    def reset(self):
        self.smc.reset()
        self.params = dict(self.base)
        self._apply()
        self.requested = dict(self.base)
        self.supervisor.reset()
        self._duty = None

    @property
    def recovering(self):
        return self.supervisor.recovering

    @property
    def recovery_events(self):
        return self.supervisor.events

    @property
    def param_array(self):
        return np.array([self.params[p] for p in PARAMS])

    def _apply(self):
        for p in PARAMS:
            setattr(self.smc, _ATTR[p], self.params[p])

    # ------------------------------------------------------------ safe box
    def clamp_params(self, params):
        """Box clamp + joint K/phi cap. Non-finite values fall back to the baseline."""
        out = {}
        for p in PARAMS:
            v = float(params[p]) if p in params else self.base[p]
            if not math.isfinite(v):
                v = self.base[p]
            lo, hi = self.ranges[p]
            out[p] = min(max(v, lo), hi)
        if out["K"] / out["phi"] > self.ratio_max:
            phi_new = out["K"] / self.ratio_max
            if phi_new <= self.ranges["phi"][1]:
                out["phi"] = phi_new
            else:                                    # phi is at its top: lower K instead
                out["phi"] = self.ranges["phi"][1]
                out["K"] = max(self.ranges["K"][0], out["phi"] * self.ratio_max)
        return out

    def target_params(self, action):
        a = np.clip(np.nan_to_num(np.asarray(action, float).reshape(-1), nan=0.0,
                                  posinf=1.0, neginf=-1.0), -1.0, 1.0)
        if a.size != 4:
            raise ValueError("adaptive-SMC action must have 4 elements")
        out = {}
        for ai, p in zip(a, PARAMS):
            lo, hi = self.ranges[p]
            p0 = self.base[p]
            out[p] = p0 * (hi / p0) ** ai if ai >= 0 else p0 * (lo / p0) ** (-ai)
        return self.clamp_params(out)

    def normalized_state(self):
        s = []
        for p in PARAMS:
            lo, hi = self.ranges[p]
            p0, v = self.base[p], self.params[p]
            if v >= p0:
                s.append(0.0 if hi == p0 else math.log(v / p0) / math.log(hi / p0))
            else:
                s.append(math.log(v / p0) / math.log(p0 / lo))
        return np.clip(s, -1.0, 1.0)

    # ----------------------------------------------------------- adaptation
    def _move_toward(self, requested):
        new = {}
        for p in PARAMS:
            cur = self.params[p]
            step = self.alpha * (math.log(requested[p]) - math.log(cur))
            step = min(max(step, -self.rate_limit), self.rate_limit)
            new[p] = cur * math.exp(step)
        self.params = self.clamp_params(new)         # box + ratio cap on every smoothed step
        self._apply()
        return dict(self.params)

    def adapt(self, action):
        self.requested = dict(self.base) if self.recovering else self.target_params(action)
        return self._move_toward(self.requested)

    def propose_params(self, **proposal):
        """Explicit proposal: clamp to the safe box, then smooth and rate-limit."""
        self.requested = dict(self.base) if self.recovering else self.clamp_params(
            {**self.params, **proposal})
        return self._move_toward(self.requested)

    def update(self, reference, measurement, dt=None, inductor_current=0.0):
        """SMC step with the current parameters; returns the duty (already saturated)."""
        dt_eff = self.smc.dt if dt is None else float(dt)
        if self.recovery_enabled and all(math.isfinite(x) for x in (reference, measurement,
                                                                    inductor_current)):
            self.supervisor.step(reference - measurement, measurement, reference,
                                 inductor_current, self._duty, dt_eff)
        duty = self.smc.update(reference, measurement, dt, inductor_current)
        self._duty = duty
        return duty
