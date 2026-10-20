"""Environment for the TD3-adaptive-SMC controller.

    state -> TD3 -> a in [-1,1]^4 -> AdaptiveSMC.adapt() -> SMC (50 us) -> duty -> converter

One RL step = 200 us; the SMC runs at the converter's 50 us step (4 updates per RL step,
parameters held), exactly like the baseline SMC, so a zero action reproduces it.

Observation (float32, shape (11,), clipped to [-5, 5]):
    0-4   standard normalised state (error, error change, integral, Vout, IL)
    5-8   current parameters (Kv, Ki, K, phi) as normalised actions in [-1, 1]
    9     last duty cycle as a normalised action
    10    baseline-recovery flag
Reward: the standard terms + param = -ASMC_W_PARAM * mean|normalised parameter state|.
"""
from __future__ import annotations

import numpy as np
from gymnasium import spaces

import config
from controllers.adaptive_smc import PARAMS, AdaptiveSMC
from environment.boost_env import REWARD_TERMS, BoostConverterEnv, duty_to_action


class AdaptiveSMCEnv(BoostConverterEnv):
    reward_term_names = REWARD_TERMS + ("param",)

    def __init__(self, recovery=None, **kwargs):
        kwargs.setdefault("include_prev_duty", False)
        super().__init__(**kwargs)
        self.controller = AdaptiveSMC(recovery=recovery)
        c = config.ENV_OBS_CLIP
        self.observation_space = spaces.Box(-c, c, shape=(11,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        self._init_state()

    def _check_action_size(self, a):
        if a.size != 4:
            raise ValueError(f"action must have exactly 4 elements, got shape {a.shape}")

    def _begin_action(self, a, invalid_action):
        if invalid_action:                      # non-finite request: hold the current parameters
            self.controller.adapt(self.controller.normalized_state())
        else:
            self.controller.adapt(a)
        return None

    def _substep_duty(self, duty_cmd):
        c = self.converter
        return self.controller.update(c.Vref, c.Vout, c.timestep, c.IL)

    def _extra_reward_terms(self):
        return {"param": -config.ASMC_W_PARAM * float(np.mean(np.abs(self.controller.normalized_state())))}

    def _on_reset(self):
        self.controller.reset()

    def _observation(self):
        base = super()._observation()
        extra = np.concatenate([self.controller.normalized_state(),
                                duty_to_action(self.prev_duty),
                                [float(self.controller.recovering)]]).astype(np.float32)
        return np.clip(np.concatenate([base, extra]), -config.ENV_OBS_CLIP,
                       config.ENV_OBS_CLIP).astype(np.float32)

    def _info(self, reward_terms):
        info = super()._info(reward_terms)
        p = self.controller.params
        info.update(Kv=p["Kv"], Ki=p["Ki"], K=p["K"], phi=p["phi"],
                    recovering=self.controller.recovering,
                    recovery_events=self.controller.recovery_events,
                    recovery_reason=self.controller.supervisor.reason)
        return info
