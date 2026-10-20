"""Gymnasium environment around the validated BoostConverter (Gymnasium >= 1.0 API).

Observation (float32, shape (5,), every entry clipped to [-5, 5]):
    0  e      = (Vref - Vout) / 10 V
    1  de     = (e_k - e_{k-1}) in volts per RL step / 1 V
    2  int_e  = clamp(integral of e dt, +-0.2 V s) / 0.1 V s     (anti-windup clamp)
    3  Vout / 24 V
    4  IL / 10 A
  Scales are chosen so that each signal is O(1) in normal operation and at most
  ~2-3 during the cold-start transient (e = 2.4, IL = 0.9, de up to ~3).
  Vin and R are NOT observed: the agent must infer them (the generalisation
  challenge).  Optional ``include_prev_duty=True`` appends the last duty as a
  6th entry (helps with the smoothness penalty; off by default = spec).

Action (float32, shape (1,)): NORMALISED a in [-1, 1], mapped linearly to the
physical duty cycle  D = 0.05 + (a + 1)/2 * 0.90  (a=-1 -> 0.05, 0 -> 0.5,
+1 -> 0.95) by ``action_to_duty``; ``duty_to_action`` is the exact inverse.
Out-of-range actions are clipped to [-1, 1] first, non-finite actions are
replaced by the last duty and penalised.  Training and evaluation both use
these same two functions.

Time: one RL step = ENV_CONTROL_STEPS converter steps of 50 us (duty held),
i.e. 200 us.  Episode length ENV_MAX_STEPS (truncation); termination only on
protection trips (over-voltage, over-current, invalid state, solver failure).

Reward per RL step = sum of the terms below (see ``reward_terms``), x = |e| / 10 V:
    err   = -(0.5 x + 0.5 sqrt(x))          tracking; the sqrt keeps a STEEP gradient
                                            near zero so 0.2 V of offset is still costly
    over  = -0.5 max(0, Vout - Vref) / 10   extra penalty for overshoot
    duty  = -0.05 (sqrt(1 + (dD/0.05)^2) - 1)   pseudo-Huber: quadratic for small
                                            jitter, only linear for legitimate big moves
    de    = -0.05 |de| / 1 V                damps ringing / limit cycles
    cur   = -1.0 (max(0, IL_max - 12 A)/12 A)^2   soft over-current penalty (sub-step peak)
    prec  = +0.5 (exp(-(e/0.5)^2) + exp(-(e/0.1)^2)) / 2   smooth precision bonus:
                                            +0.25 at e=0, still 0.1 at e = 0.2 V
    alive = +1.0                            survival bonus
  Typical range: about -3 (cold start) .. +1.5 (perfectly regulated); worst unsafe
  state about -7.  On a protection trip / invalid state the step reward is
  -fail_penalty (50): it exceeds the cost of surviving the whole episode in any
  normal operating mode, so tripping is never a good strategy, yet it is only
  ~10x a bad step so it does not swamp the critic.  Non-finite action: -1 extra.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import config
from converter import BoostConverter


def action_to_duty(action) -> float:
    """Normalised action in [-1, 1] -> physical duty in [DUTY_MIN, DUTY_MAX]."""
    a = float(np.clip(action, -1.0, 1.0))
    return config.DUTY_MIN + 0.5 * (a + 1.0) * (config.DUTY_MAX - config.DUTY_MIN)


REWARD_TERMS = ("err", "over", "duty", "de", "cur", "prec", "alive")


def reward_terms(error, d_error, duty, prev_duty, il_max, w=None):
    """Per-step reward terms (dict). Pure function so it can be unit-tested."""
    w = config.ENV_REWARD if w is None else w
    x = abs(error) / config.ENV_E_SCALE
    s1, s2 = w["prec_sigma"]
    return dict(
        err=-(w["w_lin"] * x + w["w_sqrt"] * np.sqrt(x)),
        over=-w["w_over"] * max(0.0, -error) / config.ENV_E_SCALE,
        duty=-w["w_duty"] * (np.sqrt(1.0 + ((duty - prev_duty) / w["duty_scale"]) ** 2) - 1.0),
        de=-w["w_de"] * min(abs(d_error), 5.0) / config.ENV_DE_SCALE,
        cur=-w["w_cur"] * (max(0.0, il_max - w["i_soft"]) / w["i_soft"]) ** 2,
        prec=0.5 * w["w_prec"] * (np.exp(-(error / s1) ** 2) + np.exp(-(error / s2) ** 2)),
        alive=w["alive"])


def duty_to_action(duty) -> np.ndarray:
    """Physical duty -> normalised action array of shape (1,), float32."""
    d = float(np.clip(duty, config.DUTY_MIN, config.DUTY_MAX))
    return np.array([2.0 * (d - config.DUTY_MIN) / (config.DUTY_MAX - config.DUTY_MIN) - 1.0],
                    dtype=np.float32)


class BoostConverterEnv(gym.Env):
    metadata = {"render_modes": ["ansi", "human"], "render_fps": 100}
    reward_term_names = REWARD_TERMS     # subclasses may extend (see _extra_reward_terms)
    record_trace = False                 # True: info["trace"] holds every converter sub-step

    # ---- extension hooks (the base class implements the direct-duty controller) ----
    def _check_action_size(self, a):
        if a.size != 1:
            raise ValueError(f"action must have exactly 1 element, got shape {a.shape}")

    def _begin_action(self, a, invalid_action):
        """Turn the agent action into the held duty command (base: normalised duty)."""
        return self.prev_duty if invalid_action else action_to_duty(a[0])

    def _substep_duty(self, duty_cmd):
        """Duty applied at each converter sub-step (base: the held command)."""
        return duty_cmd

    def _extra_reward_terms(self):
        return {}

    def _on_reset(self):
        """Called at the end of reset(), before the first observation is built."""

    def __init__(self, render_mode=None, max_steps=config.ENV_MAX_STEPS,
                 control_steps=config.ENV_CONTROL_STEPS,
                 vin_range=config.ENV_VIN_RANGE, r_range=config.ENV_R_RANGE,
                 vref_range=config.ENV_VREF_RANGE,
                 component_tolerance=config.ENV_COMPONENT_TOL,
                 disturbance_prob=config.ENV_DISTURBANCE_PROB,
                 disturbance_window=config.ENV_DISTURBANCE_WINDOW,
                 disturbance_kinds=config.ENV_DISTURBANCE_KINDS,
                 disturbance_vin_delta=config.ENV_DISTURBANCE_VIN_DELTA,
                 disturbance_load_ratio=config.ENV_DISTURBANCE_LOAD_RATIO,
                 disturbance_vref_delta=config.ENV_DISTURBANCE_VREF_DELTA,
                 include_prev_duty=False, reward_weights=None):
        super().__init__()
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"unsupported render_mode {render_mode!r}")
        self.render_mode = render_mode
        self.max_steps, self.control_steps = int(max_steps), int(control_steps)
        self.vin_range, self.r_range, self.vref_range = vin_range, r_range, vref_range
        self.component_tolerance = float(component_tolerance)
        self.disturbance_prob = float(disturbance_prob)
        self.disturbance_window = tuple(disturbance_window)
        self.disturbance_kinds = tuple(disturbance_kinds)
        self.disturbance_vin_delta = tuple(disturbance_vin_delta)
        self.disturbance_load_ratio = tuple(disturbance_load_ratio)
        self.disturbance_vref_delta = tuple(disturbance_vref_delta)
        bad = set(self.disturbance_kinds) - {"vin", "load", "both", "vref"}
        if bad or not self.disturbance_kinds:
            raise ValueError(f"disturbance_kinds must be a non-empty subset of vin/load/both/vref, got {bad}")
        if not 0.0 <= self.disturbance_window[0] <= self.disturbance_window[1] <= 1.0:
            raise ValueError("disturbance_window must satisfy 0 <= lo <= hi <= 1")
        self.include_prev_duty = bool(include_prev_duty)
        self.rw = dict(config.ENV_REWARD, **(reward_weights or {}))

        self.converter = BoostConverter()
        self.dt = self.converter.timestep * self.control_steps       # RL step [s]
        n_obs = 6 if self.include_prev_duty else 5
        c = config.ENV_OBS_CLIP
        self.observation_space = spaces.Box(-c, c, shape=(n_obs,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self._nominal = dict(L=self.converter.L, C=self.converter.C)
        self._disturbance = None
        self._init_state()

    # ------------------------------------------------------------ helpers
    def _init_state(self):
        self.step_count = 0
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_duty = 0.0
        self.last_obs = np.zeros(self.observation_space.shape, np.float32)
        self.episode_stats = dict(return_=0.0, terms={k: 0.0 for k in self.reward_term_names})

    def _uniform(self, lo_hi):
        return float(self.np_random.uniform(*lo_hi))

    def _observation(self):
        conv = self.converter
        e = conv.Vref - conv.Vout
        obs = [e / config.ENV_E_SCALE,
               (e - self.prev_error) / config.ENV_DE_SCALE,
               self.integral / config.ENV_INT_SCALE,
               conv.Vout / config.ENV_V_SCALE,
               conv.IL / config.ENV_I_SCALE]
        if self.include_prev_duty:
            obs.append(self.prev_duty)
        obs = np.nan_to_num(np.asarray(obs, np.float64), nan=0.0,
                            posinf=config.ENV_OBS_CLIP, neginf=-config.ENV_OBS_CLIP)
        return np.clip(obs, -config.ENV_OBS_CLIP, config.ENV_OBS_CLIP).astype(np.float32)

    def _apply(self, vin=None, load=None, vref=None):
        if vin is not None: self.converter.set_input_voltage(vin)
        if load is not None: self.converter.set_load(load)
        if vref is not None: self.converter.set_reference(vref)

    def _random_disturbance(self, vin, load, vref):
        """Random mid-episode step, relative to the episode's initial conditions.

        Magnitudes / times come from the configurable ranges; the new values are clipped to
        the operating ranges.  (The standard evaluation scenarios are fixed points that the
        continuous distribution essentially never hits.)
        """
        lo, hi = self.disturbance_window
        step = int(self.np_random.integers(int(lo * self.max_steps), max(int(hi * self.max_steps),
                                                                          int(lo * self.max_steps) + 1)))
        kind = self.disturbance_kinds[int(self.np_random.integers(len(self.disturbance_kinds)))]
        dist = dict(step=step, vin=None, load=None, vref=None, kind=kind)

        def signed_delta(base, delta_rng, bounds):
            """base +/- mag with mag ~ U(delta_rng): the direction is chosen so the FULL
            magnitude fits inside ``bounds``; only if neither direction fits is it clipped."""
            mag = self._uniform(delta_rng)
            up, down = base + mag, base - mag
            fits = [v for v in (up, down) if bounds[0] <= v <= bounds[1]]
            if fits:
                return float(fits[int(self.np_random.integers(len(fits)))])
            return float(np.clip(up if (bounds[1] - base) >= (base - bounds[0]) else down, *bounds))

        if kind in ("vin", "both"):
            dist["vin"] = signed_delta(vin, self.disturbance_vin_delta, self.vin_range)
        if kind in ("load", "both"):
            ratio = float(np.exp(self._uniform(tuple(np.log(self.disturbance_load_ratio)))))
            dist["load"] = float(np.clip(load * ratio, *self.r_range))
        if kind == "vref":
            dist["vref"] = signed_delta(vref, self.disturbance_vref_delta, self.vref_range)
        return dist

    # -------------------------------------------------------------- API
    def reset(self, *, seed=None, options=None):
        """options (all optional): vin, load, vref, init ('cold' | 'precharged' |
        'equilibrium'), disturbance = dict(step=int, vin=.., load=.., vref=..)
        or None to disable the random mid-episode disturbance."""
        super().reset(seed=seed)
        options = options or {}
        conv = self.converter
        tol = self.component_tolerance
        conv.L = self._nominal["L"] * (1 + self._uniform((-tol, tol)) if tol else 1.0)
        conv.C = self._nominal["C"] * (1 + self._uniform((-tol, tol)) if tol else 1.0)
        vin = options.get("vin", self._uniform(self.vin_range))
        load = options.get("load", self._uniform(self.r_range))
        vref = options.get("vref", self._uniform(self.vref_range))
        self._apply(vin, load, vref)

        init = options.get("init")
        if init is None:
            names = list(config.ENV_INIT_PROBS)
            init = names[self.np_random.choice(len(names), p=list(config.ENV_INIT_PROBS.values()))]
        if init == "cold":
            conv.reset(IL=0.0, Vout=0.0)
        elif init == "precharged":
            conv.reset(IL=0.0, Vout=conv.Vin)
        elif init == "equilibrium":
            d_ideal = float(np.clip(1.0 - conv.Vin / vref, config.DUTY_MIN, config.DUTY_MAX))
            il_eq, v_eq = conv.steady_state(d_ideal)
            conv.reset(IL=il_eq * self._uniform((0.5, 1.5)), Vout=v_eq * self._uniform((0.85, 1.1)))
        else:
            raise ValueError(f"unknown init {init!r}")
        self.init_mode = init

        self._init_state()
        self.prev_error = conv.Vref - conv.Vout
        self.prev_duty = float(np.clip(1.0 - conv.Vin / conv.Vref, config.DUTY_MIN, config.DUTY_MAX))

        if "disturbance" in options:
            self._disturbance = options["disturbance"]
        elif self.np_random.random() < self.disturbance_prob:
            self._disturbance = self._random_disturbance(vin, load, vref)
        else:
            self._disturbance = None

        self._on_reset()
        self.last_obs = self._observation()
        return self.last_obs, self._info(reward_terms=None)

    def step(self, action):
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        self._check_action_size(a)
        invalid_action = not bool(np.all(np.isfinite(a)))
        duty_cmd = self._begin_action(a, invalid_action)
        duty = duty_cmd

        d = self._disturbance
        if d is not None and self.step_count == d["step"]:
            self._apply(d.get("vin"), d.get("load"), d.get("vref"))

        conv, failure = self.converter, None
        il_max, il_min, v_max = -np.inf, np.inf, -np.inf     # extrema over the sub-steps
        trace = {k: [] for k in ("t", "vout", "il", "duty")} if self.record_trace else None
        try:
            for _ in range(self.control_steps):
                duty = self._substep_duty(duty_cmd)
                conv.step(duty)
                il_max, il_min = max(il_max, conv.IL), min(il_min, conv.IL)
                v_max = max(v_max, conv.Vout)
                if trace is not None:
                    for k, v in (("t", conv.time), ("vout", conv.Vout), ("il", conv.IL),
                                 ("duty", conv.duty)):
                        trace[k].append(v)
        except (FloatingPointError, ValueError, OverflowError) as exc:
            failure = f"converter failure: {exc}"
        state = conv.get_state()
        if failure is None and not np.all(np.isfinite(state)):
            failure = "non-finite state"
        elif failure is None and conv.Vout > config.ENV_V_MAX:
            failure = "over-voltage trip"
        elif failure is None and conv.IL > config.ENV_I_MAX:
            failure = "over-current trip"
        elif failure is None and (conv.Vout < -1e-6 or conv.IL < -1e-6):
            failure = "invalid (negative) state"

        self.step_count += 1
        terminated = failure is not None
        truncated = (not terminated) and self.step_count >= self.max_steps

        if terminated:
            if not np.all(np.isfinite(state)):
                conv.reset(IL=0.0, Vout=conv.Vin)            # keep the object usable
            terms = {k: 0.0 for k in self.reward_term_names}
            reward = -self.rw["fail_penalty"]
        else:
            e = conv.Vref - conv.Vout
            terms = {k: float(v) for k, v in reward_terms(
                e, e - self.prev_error, duty, self.prev_duty, il_max, self.rw).items()}
            terms.update(self._extra_reward_terms())
            reward = sum(terms.values())
        if invalid_action:
            reward -= 1.0

        e = conv.Vref - conv.Vout
        if not terminated:
            self.integral = float(np.clip(self.integral + e * self.dt,
                                          -config.ENV_INT_CLIP, config.ENV_INT_CLIP))
        self.prev_duty = duty            # before the observation: obs[5] = duty just applied
        obs = self._observation()        # de = e - prev_error uses the OLD prev_error
        self.prev_error = e if np.isfinite(e) else 0.0
        self.last_obs = obs
        st = self.episode_stats
        st["return_"] += reward
        for k, v in terms.items():
            st["terms"][k] += v

        info = self._info(reward_terms=terms)
        info.update(invalid_action=invalid_action, failure=failure,
                    IL_max=float(il_max), IL_min=float(il_min), Vout_max=float(v_max))
        if trace is not None:
            info["trace"] = {k: np.asarray(v) for k, v in trace.items()}
        if terminated or truncated:
            info["episode_stats"] = dict(return_=st["return_"], terms=dict(st["terms"]),
                                         length=self.step_count)
        if self.render_mode == "human":
            self.render()
        return obs, float(reward), terminated, truncated, info

    def _info(self, reward_terms):
        c = self.converter
        return dict(Vout=c.Vout, IL=c.IL, duty=c.duty, Vref=c.Vref, Vin=c.Vin, R=c.R,
                    mode=c.mode, time=c.time, init_mode=getattr(self, "init_mode", None),
                    reward_terms=reward_terms)

    def render(self):
        c = self.converter
        text = (f"t={c.time * 1e3:7.2f} ms  Vout={c.Vout:6.2f} V (ref {c.Vref:5.1f})  "
                f"IL={c.IL:6.2f} A  D={c.duty:.3f}  Vin={c.Vin:4.1f} V  R={c.R:4.1f} ohm  {c.mode}")
        if self.render_mode == "human":
            print(text)
        return text if self.render_mode in ("ansi", "human") else None

    def close(self):
        pass
