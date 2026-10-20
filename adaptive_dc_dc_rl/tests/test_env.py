import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from environment.boost_env import BoostConverterEnv, action_to_duty, duty_to_action


def act(duty):
    """Normalised action that maps to the given physical duty."""
    return duty_to_action(duty)


@pytest.fixture
def env():
    return BoostConverterEnv()


def test_spaces(env):
    assert isinstance(env, gym.Env)
    assert env.action_space.shape == (1,) and env.action_space.dtype == np.float32
    assert env.action_space.low[0] == -1.0 and env.action_space.high[0] == 1.0
    assert env.observation_space.shape == (5,) and env.observation_space.dtype == np.float32


def test_reset_returns_valid_obs_and_info(env):
    obs, info = env.reset(seed=0)
    assert obs.shape == (5,) and obs.dtype == np.float32
    assert env.observation_space.contains(obs)
    assert {"Vout", "IL", "Vin", "R", "Vref"} <= set(info)


def test_gymnasium_checker(env):
    from gymnasium.utils.env_checker import check_env
    check_env(env, skip_render_check=True)


def test_sb3_checker():
    from stable_baselines3.common.env_checker import check_env
    check_env(BoostConverterEnv(), warn=True)


def test_step_api_five_tuple(env):
    env.reset(seed=0)
    out = env.step(env.action_space.sample())
    assert len(out) == 5
    obs, r, term, trunc, info = out
    assert isinstance(r, float) and isinstance(term, bool) and isinstance(trunc, bool)
    assert env.observation_space.contains(obs)


def test_seeding_is_reproducible():
    a, b = BoostConverterEnv(), BoostConverterEnv()
    oa, _ = a.reset(seed=42); ob, _ = b.reset(seed=42)
    assert np.array_equal(oa, ob)
    for _ in range(50):
        a_fix = np.array([0.45], np.float32)
        ra = a.step(a_fix); rb = b.step(a_fix)
        assert np.array_equal(ra[0], rb[0]) and ra[1] == rb[1]


def test_randomisation_ranges():
    e = BoostConverterEnv()
    vins, rs, vrefs = [], [], []
    for s in range(200):
        _, info = e.reset(seed=s)
        vins.append(info["Vin"]); rs.append(info["R"]); vrefs.append(info["Vref"])
    for arr, (lo, hi) in ((vins, config.ENV_VIN_RANGE), (rs, config.ENV_R_RANGE),
                          (vrefs, config.ENV_VREF_RANGE)):
        assert min(arr) >= lo and max(arr) <= hi
        assert max(arr) - min(arr) > 0.8 * (hi - lo)       # actually spread out


def test_options_override_conditions(env):
    _, info = env.reset(seed=0, options=dict(vin=12.0, load=20.0, vref=24.0, init="cold",
                                             disturbance=None))
    assert (info["Vin"], info["R"], info["Vref"]) == (12.0, 20.0, 24.0)
    assert info["Vout"] == 0.0 and info["IL"] == 0.0


def test_truncation_at_max_steps():
    e = BoostConverterEnv(max_steps=20, disturbance_prob=0.0)
    e.reset(seed=0, options=dict(vin=12.0, load=20.0, vref=24.0, init="precharged"))
    for k in range(20):
        _, _, term, trunc, info = e.step(act(0.5))
        assert not term
        assert trunc == (k == 19)
    assert "episode_stats" in info and info["episode_stats"]["length"] == 20


def test_action_clipping_and_shapes(env):
    env.reset(seed=0)
    env.step(np.array([5.0]))                      # beyond +1 -> clipped to D_max
    assert env.converter.duty == config.DUTY_MAX
    env.step(np.array([-5.0], np.float32))
    assert env.converter.duty == config.DUTY_MIN
    env.step(0.5)                                  # scalar accepted (reshaped)
    with pytest.raises(ValueError):
        env.step(np.array([0.5, 0.5]))


def test_nan_inf_actions_are_handled(env):
    env.reset(seed=0)
    for bad in (np.nan, np.inf, -np.inf):
        obs, r, term, trunc, info = env.step(np.array([bad], np.float32))
        assert info["invalid_action"] is True
        assert np.all(np.isfinite(obs)) and np.isfinite(r)
        assert config.DUTY_MIN <= env.converter.duty <= config.DUTY_MAX


def test_random_rollouts_never_produce_nan_or_inf():
    e = BoostConverterEnv(component_tolerance=0.1)
    rng = np.random.default_rng(0)
    for ep in range(6):
        e.reset(seed=ep)
        done = False
        while not done:
            obs, r, term, trunc, info = e.step(rng.uniform(0.05, 0.95, size=1))
            assert np.all(np.isfinite(obs)) and np.isfinite(r)
            assert env_obs_ok(e, obs)
            assert e.converter.IL >= -1e-9
            done = term or trunc


def env_obs_ok(e, obs):
    return e.observation_space.contains(obs)


def test_overvoltage_terminates_with_penalty():
    e = BoostConverterEnv(disturbance_prob=0.0)
    e.reset(seed=0, options=dict(vin=14.0, load=30.0, vref=28.0, init="equilibrium"))
    e.converter.Vout = config.ENV_V_MAX + 5.0       # force a protection trip
    e.converter.IL = 1.0
    _, r, term, trunc, info = e.step(act(0.5))
    assert term and not trunc and "over-voltage" in info["failure"]
    assert r == -config.ENV_REWARD["fail_penalty"]


def test_solver_failure_is_terminal_not_exception(monkeypatch):
    e = BoostConverterEnv()
    e.reset(seed=0)
    def boom(_):
        raise FloatingPointError("injected")
    monkeypatch.setattr(e.converter, "step", boom)
    _, r, term, trunc, info = e.step(act(0.5))
    assert term and "converter failure" in info["failure"] and r < 0


def test_reward_prefers_small_error():
    kw = dict(disturbance_prob=0.0)
    opt = dict(vin=12.0, load=20.0, vref=24.0, init="equilibrium")
    e1, e2 = BoostConverterEnv(**kw), BoostConverterEnv(**kw)
    e1.reset(seed=3, options=opt); e2.reset(seed=3, options=opt)
    il_eq, v_eq = e1.converter.steady_state(0.5)
    e1.converter.reset(IL=il_eq, Vout=v_eq)
    e2.converter.reset(IL=2.4, Vout=18.0)
    e1.prev_duty = e2.prev_duty = 0.5
    _, r1, *_ = e1.step(act(0.5))
    _, r2, *_ = e2.step(act(0.5))
    assert r1 > r2


def test_reward_penalises_overshoot_more_than_undershoot():
    opt = dict(vin=12.0, load=20.0, vref=24.0, init="cold", disturbance=None)
    rs = []
    for v in (22.0, 26.0):                          # same |error|
        e = BoostConverterEnv(); e.reset(seed=0, options=opt)
        e.converter.reset(IL=2.4, Vout=v); e.prev_duty = 0.5
        _, r, *_ = e.step(act(0.5))
        rs.append(r)
    assert rs[1] < rs[0]                            # overshoot costs more


def test_reward_penalises_duty_jumps():
    opt = dict(vin=12.0, load=20.0, vref=24.0, init="equilibrium", disturbance=None)
    rs = []
    for duty in (0.5, 0.9):
        e = BoostConverterEnv(); e.reset(seed=0, options=opt)
        e.converter.reset(IL=2.4, Vout=24.0); e.prev_duty = 0.5
        _, r, _, _, info = e.step(act(duty))
        rs.append(info["reward_terms"]["duty"])
    assert rs[0] == pytest.approx(0.0, abs=1e-6) and rs[1] < -0.01


def test_integral_state_is_clamped():
    e = BoostConverterEnv(max_steps=2000, disturbance_prob=0.0)
    e.reset(seed=0, options=dict(vin=10.0, load=30.0, vref=28.0, init="cold"))
    for _ in range(1000):
        _, _, term, trunc, _ = e.step(act(0.05))   # starve it of duty
        if term or trunc:
            break
    assert abs(e.integral) <= config.ENV_INT_CLIP + 1e-12
    assert e.last_obs[2] == pytest.approx(config.ENV_INT_CLIP / config.ENV_INT_SCALE, rel=1e-3)


def test_midepisode_disturbance_is_applied():
    e = BoostConverterEnv(max_steps=30)
    e.reset(seed=0, options=dict(vin=12.0, load=20.0, vref=24.0, init="precharged",
                                 disturbance=dict(step=10, vin=10.0, load=10.0)))
    for k in range(30):
        _, _, term, _, info = e.step(act(0.5))
        if term: break
        assert (info["Vin"], info["R"]) == ((12.0, 20.0) if k < 10 else (10.0, 10.0))


def test_prev_duty_observation_option():
    e = BoostConverterEnv(include_prev_duty=True)
    assert e.observation_space.shape == (6,)
    obs, _ = e.reset(seed=0)
    obs, *_ = e.step(act(0.3))
    assert obs[5] == pytest.approx(0.3)
    assert e.observation_space.contains(obs)


def test_render_ansi():
    e = BoostConverterEnv(render_mode="ansi")
    e.reset(seed=0)
    assert "Vout" in e.render()
    with pytest.raises(ValueError):
        BoostConverterEnv(render_mode="bogus")


def test_normalisation_keeps_observations_order_one():
    """Even during the worst case (cold start, 28 V reference) |obs| stays small."""
    e = BoostConverterEnv(disturbance_prob=0.0)
    e.reset(seed=0, options=dict(vin=10.0, load=10.0, vref=28.0, init="cold"))
    peak = 0.0
    for _ in range(200):
        obs, _, term, trunc, _ = e.step(act(0.9))
        peak = max(peak, float(np.abs(obs).max()))
        if term or trunc: break
    assert peak <= config.ENV_OBS_CLIP
