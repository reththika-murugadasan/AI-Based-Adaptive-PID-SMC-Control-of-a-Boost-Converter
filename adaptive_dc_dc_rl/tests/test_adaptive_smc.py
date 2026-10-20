import hashlib
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from controllers.adaptive_smc import PARAMS, AdaptiveSMC, default_ranges
from controllers.safety_supervisor import RecoverySupervisor
from controllers.smc import SlidingModeController
from environment.adaptive_smc_env import AdaptiveSMCEnv
from experiments.run_pid import run_scenario as run_baseline_scenario

ROOT = Path(__file__).resolve().parents[1]
BASE = dict(Kv=0.65641, Ki=164.74, K=0.3, phi=0.8)
DT = config.DT
STEPS = lambda sec: int(round(sec / DT))


def settle(c, action, n=400):
    for _ in range(n):
        c.adapt(action)
    return c.params


def feed(c, err, n, il=2.4, ref=24.0):
    for _ in range(n):
        c.update(ref, ref - err, DT, il)


# ---------------------------------------------- validated baselines unchanged
def test_baseline_parameters_are_the_validated_ones():
    assert (config.SMC_KV, config.SMC_KI, config.SMC_GAIN, config.SMC_BOUNDARY) == (
        BASE["Kv"], BASE["Ki"], BASE["K"], BASE["phi"])
    s = SlidingModeController()
    assert (s.Kv, s.Ki, s.switching_gain, s.boundary_layer) == (
        BASE["Kv"], BASE["Ki"], BASE["K"], BASE["phi"])


@pytest.mark.parametrize("fname,sha", [
    ("smc.py", "cec4b304a3233a83fde8a536660bab20bcfe20741499a03c250025692b360208"),
    ("pid.py", "bd2366e3bb8c6f703c6bbd9137aa413202579ee9f56f23b4cd2e442670b71f91")])
def test_validated_baseline_controller_files_are_unmodified(fname, sha):
    """Guard: SMC (Stage 3) and PID (Stage 2) sources must not change in later stages."""
    assert hashlib.sha256((ROOT / "controllers" / fname).read_bytes()).hexdigest() == sha


# ------------------------------------------------------------ parameter bounds
def test_zero_action_requests_exactly_the_baseline():
    c = AdaptiveSMC()
    assert c.target_params(np.zeros(4)) == BASE


def test_extreme_actions_reach_the_active_range_ends():
    c = AdaptiveSMC()
    lo, hi = c.target_params(-np.ones(4)), c.target_params(np.ones(4))
    r = default_ranges()
    for p in ("Kv", "Ki", "K"):
        assert lo[p] == pytest.approx(r[p][0]) and hi[p] == pytest.approx(r[p][1])


def test_mapping_monotonic_for_each_unconstrained_parameter():
    c = AdaptiveSMC()
    a = np.linspace(-1, 1, 41)
    for i, p in enumerate(("Kv", "Ki")):
        vals = [c.target_params(np.eye(4)[i] * x)[p] for x in a]
        assert all(b >= a_ - 1e-15 for a_, b in zip(vals, vals[1:]))


def test_active_box_is_tight_and_inside_the_certified_box():
    f, cert = config.ASMC_RANGE_FACTORS, config.ASMC_CERTIFIED_FACTORS
    for p in PARAMS:
        assert f[p][0] <= 1.0 <= f[p][1]
        assert cert[p][0] <= f[p][0] and f[p][1] <= cert[p][1]
        assert 0.8 <= f[p][0] and f[p][1] <= 1.25                 # at most +-25 % anywhere
    assert config.ASMC_MAX_RATIO_FACTOR <= 1.2


def test_ranges_outside_the_certified_box_are_refused():
    wide = {p: tuple(BASE[p] * x for x in (0.5, 2.0)) for p in PARAMS}
    with pytest.raises(ValueError, match="certified"):
        AdaptiveSMC(ranges=wide)
    c = AdaptiveSMC(ranges=wide, allow_uncertified=True)
    assert c.ranges["Kv"][1] == pytest.approx(BASE["Kv"] * 2.0)
    with pytest.raises(ValueError):
        AdaptiveSMC(max_ratio_factor=2.0)


def test_baseline_must_lie_inside_ranges():
    bad = default_ranges(); bad["K"] = (0.31, 0.33)
    with pytest.raises(ValueError):
        AdaptiveSMC(ranges=bad)


@pytest.mark.parametrize("action", [np.full(4, 50.0), np.full(4, -50.0), np.array([np.nan, np.inf, -np.inf, np.nan]),
                                    np.array([1.0, -1.0, 1.0, -1.0]), np.array([-1.0, 1.0, 1.0, -1.0])])
def test_params_stay_in_bounds_for_extreme_or_invalid_actions(action):
    c = AdaptiveSMC()
    settle(c, action)
    for p in PARAMS:
        lo, hi = c.ranges[p]
        assert lo - 1e-15 <= c.params[p] <= hi + 1e-15 and math.isfinite(c.params[p])
    assert c.params["K"] / c.params["phi"] <= c.ratio_max * (1 + 1e-12)


def test_params_and_ratio_cap_hold_for_random_actions():
    c = AdaptiveSMC()
    rng = np.random.default_rng(0)
    for _ in range(5000):
        c.adapt(rng.uniform(-3, 3, size=4))
        for p in PARAMS:
            lo, hi = c.ranges[p]
            assert lo - 1e-15 <= c.params[p] <= hi + 1e-15
        assert c.params["K"] / c.params["phi"] <= c.ratio_max * (1 + 1e-12)


def test_joint_ratio_cap_is_enforced_where_the_box_alone_would_allow_it():
    c = AdaptiveSMC()
    r = c.ranges
    assert r["K"][1] / r["phi"][0] > c.ratio_max                  # the box corner violates the cap
    out = c.clamp_params(dict(Kv=BASE["Kv"], Ki=BASE["Ki"], K=r["K"][1], phi=r["phi"][0]))
    assert out["K"] / out["phi"] <= c.ratio_max * (1 + 1e-12)
    assert r["phi"][0] <= out["phi"] <= r["phi"][1]
    # discrete current-loop gain stays below 1 at 40 V (instability at 2)
    assert (out["K"] / out["phi"]) * config.DT * 40.0 / config.L < 1.0


def test_pid_values_in_the_smc_object_match_the_bounded_params():
    c = AdaptiveSMC()
    rng = np.random.default_rng(1)
    for _ in range(2000):
        c.adapt(rng.uniform(-9, 9, size=4))
        s = c.smc
        assert (s.Kv, s.Ki, s.switching_gain, s.boundary_layer) == tuple(c.params[p] for p in PARAMS)


def test_unsafe_proposals_are_clamped():
    c = AdaptiveSMC()
    out = c.clamp_params(dict(Kv=1e6, Ki=-5.0, K=float("nan"), phi=1e-9))
    assert out["Kv"] == c.ranges["Kv"][1] and out["Ki"] == c.ranges["Ki"][0]
    assert out["K"] == BASE["K"] or out["K"] <= c.ranges["K"][1]
    for p in PARAMS:
        assert c.ranges[p][0] <= out[p] <= c.ranges[p][1]
    assert out["K"] / out["phi"] <= c.ratio_max * (1 + 1e-12)


def test_propose_params_never_applies_an_unsafe_set():
    c = AdaptiveSMC()
    for _ in range(800):
        c.propose_params(Kv=1e9, Ki=1e-9, K=1e3, phi=1e-9)
        for p in PARAMS:
            lo, hi = c.ranges[p]
            assert lo - 1e-15 <= c.params[p] <= hi + 1e-15
        assert c.params["K"] / c.params["phi"] <= c.ratio_max * (1 + 1e-12)
    assert c.params["Kv"] == pytest.approx(c.ranges["Kv"][1], rel=1e-6)
    assert c.params["Ki"] == pytest.approx(c.ranges["Ki"][0], rel=1e-6)


# --------------------------------------------------------------- smoothing
def test_parameter_changes_are_rate_limited_even_with_bang_bang_requests():
    c = AdaptiveSMC()
    prev = c.param_array
    rng = np.random.default_rng(2)
    for _ in range(3000):
        c.adapt(rng.choice([-1.0, 1.0], size=4))
        now = c.param_array
        assert np.all(np.abs(np.log(now / prev)) <= config.ASMC_RATE_LIMIT + 1e-12)
        prev = now


def test_parameter_moves_gradually_then_converges():
    c = AdaptiveSMC()
    c.adapt(np.array([1.0, 0.0, 0.0, 0.0]))
    first = c.params["Kv"]
    assert BASE["Kv"] < first <= BASE["Kv"] * math.exp(config.ASMC_RATE_LIMIT) * (1 + 1e-12)
    settle(c, np.array([1.0, 0.0, 0.0, 0.0]), n=500)
    assert c.params["Kv"] == pytest.approx(c.ranges["Kv"][1], rel=1e-3)


def test_alternating_requests_are_filtered():
    c = AdaptiveSMC()
    vals = []
    for k in range(400):
        c.adapt(np.array([1.0 if k % 2 else -1.0, 0, 0, 0])); vals.append(c.params["Kv"])
    swing = max(vals[100:]) / min(vals[100:])
    full = c.ranges["Kv"][1] / c.ranges["Kv"][0]
    assert swing < 1.0 + 0.5 * (full - 1)


# ------------------------------------------------------ supervisor / recovery
def test_recovery_engages_on_persistent_large_error_but_not_on_a_spike():
    c = AdaptiveSMC()
    for _ in range(5):
        feed(c, 8.0, STEPS(1e-3)); feed(c, 0.0, STEPS(10e-3))
    assert not c.recovering                                     # short spikes: ignored
    feed(c, 6.0, STEPS(config.ASMC_REC_PERSIST + 5 * config.ASMC_REC_ERR_TAU))
    assert c.recovering and c.recovery_events == 1
    assert c.supervisor.reason == "error"


def test_recovery_engages_immediately_on_unsafe_current_or_voltage():
    c = AdaptiveSMC()
    c.update(24.0, 24.0, DT, config.ASMC_REC_IL_WARN + 1.0)
    assert c.recovering and c.supervisor.reason == "unsafe"
    c2 = AdaptiveSMC()
    c2.update(24.0, 24.0 * config.ASMC_REC_V_WARN_RATIO + 0.5, DT, 2.0)
    assert c2.recovering


def test_baseline_inrush_and_overshoot_do_not_trip_the_warnings():
    assert config.ASMC_REC_IL_WARN > 10.5          # worst baseline IL over the 9 extreme plants
    assert config.ASMC_REC_V_WARN_RATIO > 1.026    # nominal baseline cold-start overshoot


def test_supervisor_detects_persistent_chatter():
    from controllers.adaptive_smc import supervisor_config
    s = RecoverySupervisor(supervisor_config())
    d = 0.3
    for k in range(STEPS(config.ASMC_REC_CHATTER_PERSIST * 0.5)):    # too short
        d = 0.7 - d + 0.0                                           # flips +-0.4 each step
        s.step(0.0, 24.0, 24.0, 2.0, d, DT)
    assert not s.recovering
    for k in range(STEPS(config.ASMC_REC_CHATTER_PERSIST * 3)):
        d = 0.7 - d
        s.step(0.0, 24.0, 24.0, 2.0, d, DT)
    assert s.recovering and s.reason == "chatter"


def test_normal_duty_activity_does_not_look_like_chatter():
    from controllers.adaptive_smc import supervisor_config
    s = RecoverySupervisor(supervisor_config())
    for k in range(STEPS(0.1)):
        s.step(0.0, 24.0, 24.0, 2.0, 0.5 + 1e-4 * (-1) ** k, DT)
    assert not s.recovering


def test_recovery_returns_parameters_to_baseline_smoothly_and_ignores_the_agent():
    c = AdaptiveSMC()
    corner = np.array([1.0, -1.0, 1.0, -1.0])
    settle(c, corner, n=300)
    assert not np.allclose(c.param_array, list(BASE.values()))
    feed(c, 6.0, STEPS(0.05))
    assert c.recovering
    prev = c.param_array
    for _ in range(300):
        c.adapt(corner)                                        # the agent keeps asking
        now = c.param_array
        assert np.all(np.abs(np.log(now / prev)) <= config.ASMC_RATE_LIMIT + 1e-12)   # smooth
        prev = now
    assert np.allclose(c.param_array, list(BASE.values()), rtol=1e-3)


def test_recovery_releases_and_the_agent_regains_control():
    c = AdaptiveSMC()
    feed(c, 6.0, STEPS(0.05))
    assert c.recovering
    feed(c, 0.0, STEPS(config.ASMC_REC_MIN_HOLD * 0.3))
    assert c.recovering                                        # minimum hold time
    feed(c, 0.0, STEPS(0.05))
    assert not c.recovering
    c.adapt(np.array([1.0, 0, 0, 0]))
    assert c.params["Kv"] > BASE["Kv"]


def test_recovery_can_be_disabled():
    c = AdaptiveSMC(recovery=False)
    feed(c, 20.0, STEPS(0.1), il=50.0)
    assert not c.recovering
    c.adapt(np.array([1.0, 0, 0, 0]))
    assert c.params["Kv"] > BASE["Kv"]


def test_zero_action_stays_exactly_baseline_even_while_recovering():
    c = AdaptiveSMC()
    feed(c, 6.0, STEPS(0.05))
    settle(c, np.zeros(4), n=50)
    assert c.recovering and c.params == BASE


# --------------------------------------------------- duty / current / finite
def test_duty_within_limits_with_extreme_inputs_and_parameters():
    c = AdaptiveSMC()
    rng = np.random.default_rng(3)
    for k in range(6000):
        if k % 10 == 0:
            c.adapt(rng.uniform(-3, 3, size=4))
        d = c.update(rng.uniform(0, 60), rng.uniform(-5, 90), DT, rng.uniform(-5, 40))
        assert config.DUTY_MIN <= d <= config.DUTY_MAX and math.isfinite(d)


def test_non_finite_measurement_rejected():
    with pytest.raises(ValueError):
        AdaptiveSMC().update(24.0, float("nan"), DT, 2.0)


def test_current_reference_limit_is_preserved():
    c = AdaptiveSMC()
    settle(c, np.ones(4))
    for _ in range(2000):
        c.update(28.0, 0.0, DT, 0.0)
    assert c.smc.last["i_ref"] <= c.smc.i_max + 1e-12


def test_zero_action_reproduces_the_baseline_smc_exactly():
    """a = 0 must give the validated SMC trajectory bit-for-bit (50 us resolution)."""
    for kind, val in ((None, None), ("vin", 10.0), ("load", 30.0)):
        ref = run_baseline_scenario(kind, val, controller=SlidingModeController(),
                                    t_end=0.05, t_dist=0.025)
        env = AdaptiveSMCEnv(max_steps=250, disturbance_prob=0.0); env.record_trace = True
        dist = None if kind is None else dict(step=125, vin=val if kind == "vin" else None,
                                              load=val if kind == "load" else None)
        env.reset(seed=0, options=dict(vin=12.0, load=20.0, vref=24.0, init="cold", disturbance=dist))
        v, d = [], []
        for _ in range(250):
            _, _, term, trunc, info = env.step(np.zeros(4, np.float32))
            v.extend(info["trace"]["vout"]); d.extend(info["trace"]["duty"])
            if term or trunc:
                break
        assert np.array_equal(np.array(v), ref["vout"]) and np.array_equal(np.array(d), ref["duty"])


# --------------------------------------------------------------- environment
@pytest.fixture
def env():
    return AdaptiveSMCEnv()


def test_spaces_and_reset(env):
    assert env.action_space.shape == (4,) and env.action_space.low.min() == -1
    assert env.action_space.high.max() == 1
    assert env.observation_space.shape == (11,)
    obs, info = env.reset(seed=0)
    assert obs.shape == (11,) and env.observation_space.contains(obs)
    assert np.allclose(obs[5:9], 0.0) and obs[10] == 0.0
    assert (info["Kv"], info["K"], info["phi"]) == (BASE["Kv"], BASE["K"], BASE["phi"])


def test_gymnasium_and_sb3_checkers():
    from gymnasium.utils.env_checker import check_env as gcheck
    from stable_baselines3.common.env_checker import check_env as scheck
    gcheck(AdaptiveSMCEnv(), skip_render_check=True)
    scheck(AdaptiveSMCEnv(), warn=True)


def test_action_shape_validated(env):
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(np.array([0.1, 0.1, 0.1]))


def test_random_and_extreme_rollouts_finite_valid_and_safe():
    e = AdaptiveSMCEnv(component_tolerance=0.1)
    rng = np.random.default_rng(4)
    for ep in range(5):
        e.reset(seed=ep)
        done = False
        while not done:
            a = rng.choice([-1.0, 1.0], size=4) if ep % 2 else rng.uniform(-3, 3, size=4)
            obs, r, term, trunc, info = e.step(a.astype(np.float32))
            assert np.all(np.isfinite(obs)) and np.isfinite(r) and e.observation_space.contains(obs)
            assert config.DUTY_MIN <= info["duty"] <= config.DUTY_MAX
            assert info["IL_min"] >= -1e-9
            for p in PARAMS:
                lo, hi = e.controller.ranges[p]
                assert lo - 1e-15 <= info[p] <= hi + 1e-15
            assert info["K"] / info["phi"] <= e.controller.ratio_max * (1 + 1e-12)
            done = term or trunc


def test_nan_action_holds_parameters_and_is_penalised(env):
    env.reset(seed=0, options=dict(vin=12.0, load=20.0, vref=24.0, init="precharged", disturbance=None))
    env.step(np.array([0.5, -0.5, 0.2, 0.1], np.float32))
    before = env.controller.param_array.copy()
    _, r, term, _, info = env.step(np.full(4, np.nan, np.float32))
    assert info["invalid_action"] and np.isfinite(r) and not term
    assert np.allclose(env.controller.param_array, before, rtol=1e-12)


def test_protection_trip_still_terminates(env):
    env.reset(seed=0, options=dict(vin=12.0, load=20.0, vref=24.0, init="equilibrium", disturbance=None))
    env.converter.Vout = config.ENV_V_MAX + 5.0
    _, r, term, _, info = env.step(np.zeros(4, np.float32))
    assert term and "over-voltage" in info["failure"] and r == -config.ENV_REWARD["fail_penalty"]


def test_param_reward_term_zero_at_baseline_negative_away():
    e = AdaptiveSMCEnv(recovery=False)
    e.reset(seed=0, options=dict(vin=12.0, load=20.0, vref=24.0, init="equilibrium", disturbance=None))
    _, _, _, _, info = e.step(np.zeros(4, np.float32))
    assert info["reward_terms"]["param"] == 0.0
    for _ in range(80):
        _, _, _, _, info = e.step(np.array([1.0, -1.0, 1.0, 1.0], np.float32))
    assert info["reward_terms"]["param"] < -0.01
    assert set(e.episode_stats["terms"]) == set(AdaptiveSMCEnv.reward_term_names)


def test_env_reports_and_observes_recovery():
    e = AdaptiveSMCEnv(disturbance_prob=0.0)
    e.reset(seed=0, options=dict(vin=14.0, load=30.0, vref=20.0, init="cold", disturbance=None))
    seen = False
    for _ in range(400):
        obs, r, term, trunc, info = e.step(np.array([1.0, 1.0, -1.0, 1.0], np.float32))
        assert obs[10] == float(info["recovering"])
        seen |= info["recovering"]
        assert np.all(np.isfinite(obs)) and np.isfinite(r)
        if term or trunc:
            break
    assert seen          # Vin 14 -> Vref 20 cold start overshoots past 1.15 Vref: supervisor engages


def test_episode_stats_sum_to_return():
    e = AdaptiveSMCEnv(max_steps=40, disturbance_prob=0.0)
    e.reset(seed=0)
    for _ in range(40):
        _, _, term, trunc, info = e.step(np.zeros(4, np.float32))
        if term or trunc:
            break
    st = info["episode_stats"]
    assert st["return_"] == pytest.approx(sum(st["terms"].values()), abs=1e-6)


# ---------------------------------------- tiny model: train / load / evaluate
@pytest.fixture(scope="module")
def tiny_model_path(tmp_path_factory):
    from training import train_td3_adaptive_smc
    out = tmp_path_factory.mktemp("models_asmc")
    old = config.MODELS_DIR
    config.MODELS_DIR = out
    try:
        path = train_td3_adaptive_smc.train(total_timesteps=300, seed=0, model_name="tiny_asmc",
                                            learning_starts=100, tensorboard=False)
    finally:
        config.MODELS_DIR = old
    return Path(path)


def test_model_saved_and_loadable(tiny_model_path):
    from training.evaluate_td3_adaptive_smc import load_model
    assert tiny_model_path.exists() and tiny_model_path.with_name("tiny_asmc_info.json").exists()
    m = load_model(tiny_model_path)
    obs, _ = AdaptiveSMCEnv().reset(seed=0)
    a, _ = m.predict(obs, deterministic=True)
    assert a.shape == (4,) and np.all(np.abs(a) <= 1 + 1e-6)


def test_missing_model_raises():
    from training.evaluate_td3_adaptive_smc import load_model
    with pytest.raises(FileNotFoundError):
        load_model("no_such_adaptive_smc_model")


def test_deterministic_evaluation(tiny_model_path):
    from training.evaluate_td3_adaptive_smc import load_model, run_scenario
    m = load_model(tiny_model_path)
    r1 = run_scenario(m, "vin", 10.0, t_dist=0.01, t_end=0.02)
    r2 = run_scenario(m, "vin", 10.0, t_dist=0.01, t_end=0.02)
    for k in ("vout", "il", "duty") + PARAMS:
        assert np.array_equal(r1[k], r2[k])


def test_evaluation_finite_and_safe(tiny_model_path):
    from training.evaluate_td3_adaptive_smc import evaluate, load_model
    m = load_model(tiny_model_path)
    results, df = evaluate(m, t_dist=0.01, t_end=0.02)
    assert list(df.index) == list(config.PID_SCENARIOS) and df["all_finite"].all()
    need = {"rise_time_ms", "settling_time_ms", "overshoot_pct", "undershoot_V", "ss_error_V",
            "rmse_V", "max_dev_after_V", "recovery_time_ms", "duty_variation", "max_IL_A", "min_IL_A"}
    assert need <= set(df.columns)
    for r in results.values():
        assert r["duty"].min() >= config.DUTY_MIN - 1e-12 and r["duty"].max() <= config.DUTY_MAX + 1e-12
        assert np.all(np.isfinite(r["vout"])) and r["il_min"].min() >= -1e-9
        for p in PARAMS:
            lo, hi = default_ranges()[p]
            assert r[p].min() >= lo - 1e-15 and r[p].max() <= hi + 1e-15


def test_zero_policy_evaluation_equals_the_baseline_smc():
    from training.evaluate_td3_adaptive_smc import run_scenario
    ref = run_baseline_scenario("vin", 10.0, controller=SlidingModeController())
    r = run_scenario(None, "vin", 10.0)
    assert np.array_equal(r["vout"], ref["vout"]) and np.array_equal(r["duty"], ref["duty"])
