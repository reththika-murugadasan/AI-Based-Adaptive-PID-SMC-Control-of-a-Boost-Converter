"""Environment for the TD3-adaptive-PID controller.

    state -> TD3 -> a in [-1,1]^3 -> AdaptivePID.adapt() -> PID (50 us) -> duty -> converter

One RL step = 200 us.  The PID keeps running at the converter's 50 us step (4 PID
updates per RL step with the gains held), exactly like the baseline PID, so a
zero action reproduces the baseline PID trajectory.

Observation (float32, shape (9,), clipped to [-5, 5]):
    0-4  the standard normalised state (error, error change, integral, Vout, IL)
    5-7  current gains as normalised actions in [-1, 1] (0 = baseline gains) - the
         agent must know where it is inside the safe range to adapt safely
    8    last duty cycle as a normalised action in [-1, 1]
    9    baseline-recovery flag (1 while the supervisor ignores the agent's requests)
Reward: the standard terms plus  gain = -ADAPT_W_GAIN * mean|normalised gain state|
(a small pull toward the validated baseline when adaptation has no benefit).
"""
from __future__ import annotations

import numpy as np
from gymnasium import spaces

import config
from controllers.adaptive_pid import AdaptivePID
from environment.boost_env import REWARD_TERMS, BoostConverterEnv, duty_to_action


class AdaptivePIDEnv(BoostConverterEnv):
    reward_term_names = REWARD_TERMS + ("gain",)

    def __init__(self, recovery=None, **kwargs):
        kwargs.setdefault("include_prev_duty", False)
        super().__init__(**kwargs)
        self.controller = AdaptivePID(dt=self.converter.timestep, recovery=recovery)
        c = config.ENV_OBS_CLIP
        self.observation_space = spaces.Box(-c, c, shape=(10,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self._init_state()

    # ----------------------------------------------------------------- hooks
    def _check_action_size(self, a):
        if a.size != 3:
            raise ValueError(f"action must have exactly 3 elements, got shape {a.shape}")

    def _begin_action(self, a, invalid_action):
        if invalid_action:                      # non-finite request: hold the current gains
            self.controller.adapt(self.controller.normalized_state())
        else:
            self.controller.adapt(a)
        return None

    def _substep_duty(self, duty_cmd):
        c = self.converter
        return self.controller.update(c.Vref, c.Vout, c.timestep, c.IL)

    def _extra_reward_terms(self):
        return {"gain": -config.ADAPT_W_GAIN * float(np.mean(np.abs(self.controller.normalized_state())))}

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
        info.update(Kp=self.controller.gains["Kp"], Ki=self.controller.gains["Ki"],
                    Kd=self.controller.gains["Kd"], recovering=self.controller.recovering,
                    recovery_events=self.controller.recovery_events)
        return info
