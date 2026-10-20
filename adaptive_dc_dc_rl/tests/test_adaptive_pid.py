import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from controllers.adaptive_pid import GAINS, AdaptivePID
from controllers.pid import PIDController
from environment.adaptive_pid_env import AdaptivePIDEnv
from experiments.run_pid import run_scenario as run_pid_scenario

BASE = (config.PID_KP, config.PID_KI, config.PID_KD)
RANGES = (config.ADAPT_KP_RANGE, config.ADAPT_KI_RANGE, config.ADAPT_KD_RANGE)


def settle(ctrl, action, n=400):
    for _ in range(n):
        ctrl.adapt(action)
    return ctrl.gains


# ------------------------------------------------------------- gain bounds
def test_baseline_pid_gains_are_unchanged():
    assert BASE == (0.0049154, 6.6994, 2.7364e-5)
    assert PIDController().Kp == config.PID_KP


def test_zero_action_requests_exactly_the_baseline():
    c = AdaptivePID()
    t = c.target_gains(np.zeros(3))
    assert [t[g] for g in GAINS] == list(BASE)


def test_extreme_actions_hit_the_range_ends():
    c = AdaptivePID()
    lo = c.target_gains(-np.ones(3)); hi = c.target_gains(np.ones(3))
    for g, (l, h) in zip(GAINS, RANGES):
        assert lo[g] == pytest.approx(l) and hi[g] == pytest.approx(h)


def test_mapping_monotonic_in_each_gain():
    c = AdaptivePID()
    a = np.linspace(-1, 1, 41)
    for i, g in enumerate(GAINS):
        vals = [c.target_gains(np.eye(3)[i] * x)[g] for x in a]
        assert all(b >= a_ - 1e-15 for a_, b in zip(vals, vals[1:]))


def test_ranges_must_contain_baseline():
    bad = ((0.01, 0.02), RANGES[1], RANGES[2])        # Kp baseline 0.0049 outside
    with pytest.raises(ValueError):
        AdaptivePID(ranges=bad)


@pytest.mark.parametrize("action", [np.full(3, 50.0), np.full(3, -50.0),
                                    np.array([np.nan, np.inf, -np.inf])])
def test_gains_stay_in_bounds_for_extreme_or_invalid_actions(action):
    c = AdaptivePID()
    settle(c, action)
    for g, (lo, hi) in zip(GAINS, RANGES):
        assert lo - 1e-15 <= c.gains[g] <= hi + 1e-15 and math.isfinite(c.gains[g])


def test_gains_stay_in_bounds_for_random_action_sequences():
    c = AdaptivePID()
    rng = np.random.default_rng(0)
    for _ in range(5000):
        c.adapt(rng.uniform(-3, 3, size=3))
        for g, (lo, hi) in zip(GAINS, RANGES):
            assert lo - 1e-15 <= c.gains[g] <= hi + 1e-15


def test_range_factors_match_the_margin_analysis():
    f, cert = config.ADAPT_RANGE_FACTORS, config.ADAPT_CERTIFIED_FACTORS
    for g in GAINS:
        assert f[g][0] <= 1.0 <= f[g][1]                       # baseline inside every range
        assert cert[g][0] <= f[g][0] and f[g][1] <= cert[g][1]  # active inside certified
    # the rejected wide boxes must not be in use
    assert f["Ki"][1] <= 1.05 and f["Ki"][0] >= 0.7             # Ki x1.5 leaves ~18 deg, x2 unstable
    assert f["Kp"][1] <= 1.5 and f["Kd"][0] >= 0.9 and f["Kd"][1] <= 1.2
    assert max(max(v) for v in f.values()) < 1.5 + 1e-12 and min(min(v) for v in f.values()) >= 0.7


def test_ranges_outside_the_certified_box_are_refused():
    base = np.array(BASE)
    wide = tuple(tuple(b * x for x in (0.25, 2.0)) for b in base)      # r=2-style box
    with pytest.raises(ValueError, match="certified"):
        AdaptivePID(ranges=wide)
    c = AdaptivePID(ranges=wide, allow_uncertified=True)               # explicit override only
    assert c.ranges["Kp"][1] == pytest.approx(BASE[0] * 2.0)


def test_narrower_than_certified_ranges_are_accepted():
    base = np.array(BASE)
    narrow = tuple((b * 0.95, b * 1.05) for b in base)
    c = AdaptivePID(ranges=narrow)
    settle(c, np.ones(3))
    for g, b in zip(GAINS, base):
        assert c.gains[g] == pytest.approx(b * 1.05, rel=1e-3)


# ----------------------------------------------------------- gain smoothing
def test_gain_changes_are_rate_limited():
    c = AdaptivePID()
    prev = c.gain_array
    rng = np.random.default_rng(1)
    for _ in range(2000):
        c.adapt(rng.choice([-1.0, 1.0], size=3))      # worst case: bang-bang requests
        now = c.gain_array
        assert np.all(np.abs(np.log(now / prev)) <= config.ADAPT_RATE_LIMIT + 1e-12)
        prev = now


def test_gain_moves_gradually_then_converges():
    c = AdaptivePID()
    c.adapt(np.array([1.0, 0.0, 0.0]))
    first = c.gains["Kp"]
    assert config.PID_KP < first <= config.PID_KP * math.exp(config.ADAPT_RATE_LIMIT) * (1 + 1e-12)
    settle(c, np.array([1.0, 0.0, 0.0]), n=500)
    assert c.gains["Kp"] == pytest.approx(config.ADAPT_KP_RANGE[1], rel=1e-3)


def test_alternating_requests_are_filtered():
    """+1/-1 flipping every step must move the gain much less than the full swing."""
    c = AdaptivePID()
    ks = []
    for k in range(400):
        c.adapt(np.array([1.0 if k % 2 else -1.0, 0, 0])); ks.append(c.gains["Kp"])
    swing = max(ks[100:]) / min(ks[100:])
    full = config.ADAPT_KP_RANGE[1] / config.ADAPT_KP_RANGE[0]
    assert swing < 1.0 + 0.25 * (full - 1) and full > 2


def test_integral_state_unchanged_when_gain_changes():
    c = AdaptivePID()
    for _ in range(200):
        c.update(24.0, 22.0, 5e-5)
    before = c.pid.i_term
    c.adapt(np.array([0.0, -1.0, 0.0]))
    assert c.pid.i_term == before                 # bumpless (stored in duty units)


# ------------------------------------------------ PID output / safety limits
def test_pid_output_within_duty_limits_for_extreme_inputs():
    c = AdaptivePID()
    rng = np.random.default_rng(2)
    for _ in range(5000):
        c.adapt(rng.uniform(-1, 1, size=3))
        d = c.update(rng.uniform(0, 60), rng.uniform(-10, 120), 5e-5)
        assert config.DUTY_MIN <= d <= config.DUTY_MAX and math.isfinite(d)


def test_non_finite_measurement_rejected():
    with pytest.raises(ValueError):
        AdaptivePID().update(24.0, float("nan"), 5e-5)


def test_zero_action_reproduces_baseline_pid_exactly():
    """a = 0 must give the validated PID trajectory bit-for-bit (50 us resolution)."""
    for kind, val in ((None, None), ("vin", 10.0), ("load", 30.0)):
        ref = run_pid_scenario(kind, val, t_end=0.05, t_dist=0.025)
        env = AdaptivePIDEnv(max_steps=250, disturbance_prob=0.0); env.record_trace = True
        dist = None if kind is None else dict(step=125, vin=val if kind == "vin" else None,
                                              load=val if kind == "load" else None)
        env.reset(seed=0, options=dict(vin=12.0, load=20.0, vref=24.0, init="cold",
                                       disturbance=dist))
        v = []
        for _ in range(250):
            _, _, term, trunc, info = env.step(np.zeros(3, np.float32))
            v.extend(info["trace"]["vout"])
            if term or trunc:
                break
        assert np.array_equal(np.array(v), ref["vout"])


# ------------------------------------------------------------- environment
@pytest.fixture
def env():
    return AdaptivePIDEnv()


def test_spaces_and_reset(env):
    assert env.action_space.shape == (3,) and env.action_space.low.min() == -1
    assert env.action_space.high.max() == 1
    assert env.observation_space.shape == (10,)
    obs, info = env.reset(seed=0)
    assert obs.shape == (10,) and env.observation_space.contains(obs)
    assert np.allclose(obs[5:8], 0.0) and info["Kp"] == config.PID_KP   # starts at baseline


def test_gymnasium_and_sb3_checkers():
    from gymnasium.utils.env_checker import check_env as gcheck
    from stable_baselines3.common.env_checker import check_env as scheck
    gcheck(AdaptivePIDEnv(), skip_render_check=True)
    scheck(AdaptivePIDEnv(), warn=True)


def test_action_shape_validated(env):
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(np.array([0.1]))


def test_random_and_extreme_rollouts_finite_with_valid_duty():
    e = AdaptivePIDEnv(component_tolerance=0.1)
    rng = np.random.default_rng(3)
    for ep in range(5):
        e.reset(seed=ep)
        done = False
        while not done:
            a = rng.choice([-1.0, 1.0], size=3) if ep % 2 else rng.uniform(-3, 3, size=3)
            obs, r, term, trunc, info = e.step(a.astype(np.float32))
            assert np.all(np.isfinite(obs)) and np.isfinite(r)
            assert e.observation_space.contains(obs)
            assert config.DUTY_MIN <= info["duty"] <= config.DUTY_MAX
            assert info["IL_min"] >= -1e-9
            for g, (lo, hi) in zip(GAINS, RANGES):
                assert lo - 1e-15 <= info[g] <= hi + 1e-15
            done = term or trunc


def test_nan_action_holds_gains_and_is_penalised(env):
    env.reset(seed=0, options=dict(vin=12.0, load=20.0, vref=24.0, init="precharged",
                                   disturbance=None))
    env.step(np.array([0.5, -0.5, 0.2], np.float32))
    k_before = env.controller.gain_array.copy()
    _, r, term, _, info = env.step(np.array([np.nan, np.nan, np.nan], np.float32))
    assert info["invalid_action"] and np.isfinite(r) and not term
    assert np.allclose(env.controller.gain_array, k_before, rtol=1e-12)


def test_protection_trip_still_terminates(env):
    env.reset(seed=0, options=dict(vin=12.0, load=20.0, vref=24.0, init="equilibrium",
                                   disturbance=None))
    env.converter.Vout = config.ENV_V_MAX + 5.0
    _, r, term, _, info = env.step(np.zeros(3, np.float32))
    assert term and "over-voltage" in info["failure"]
    assert r == -config.ENV_REWARD["fail_penalty"]


def test_gain_reward_term_zero_at_baseline_and_negative_away():
    env = AdaptivePIDEnv(recovery=False)          # isolate the gain term from the supervisor
    env.reset(seed=0, options=dict(vin=12.0, load=20.0, vref=24.0, init="equilibrium",
                                   disturbance=None))
    _, _, _, _, info = env.step(np.zeros(3, np.float32))
    assert info["reward_terms"]["gain"] == 0.0
    for _ in range(80):
        _, _, _, _, info = env.step(np.array([-1.0, -1.0, 1.0], np.float32))
    assert info["reward_terms"]["gain"] < -0.01
    assert env.episode_stats["terms"].keys() == set(AdaptivePIDEnv.reward_term_names)


def test_episode_stats_sum_to_return():
    e = AdaptivePIDEnv(max_steps=40, disturbance_prob=0.0)
    e.reset(seed=0)
    for _ in range(40):
        _, _, term, trunc, info = e.step(np.zeros(3, np.float32))
        if term or trunc:
            break
    st = info["episode_stats"]
    assert st["return_"] == pytest.approx(sum(st["terms"].values()), abs=1e-6)


# -------------------------------------------- tiny model: train/load/evaluate
@pytest.fixture(scope="module")
def tiny_model_path(tmp_path_factory):
    from training import train_td3_adaptive_pid
    out = tmp_path_factory.mktemp("models_apid")
    old = config.MODELS_DIR
    config.MODELS_DIR = out
    try:
        path = train_td3_adaptive_pid.train(total_timesteps=300, seed=0, model_name="tiny_apid",
                                            learning_starts=100, tensorboard=False)
    finally:
        config.MODELS_DIR = old
    return Path(path)


def test_model_saved_and_loadable(tiny_model_path):
    from training.evaluate_td3_adaptive_pid import load_model
    assert tiny_model_path.exists() and tiny_model_path.with_name("tiny_apid_info.json").exists()
    m = load_model(tiny_model_path)
    obs, _ = AdaptivePIDEnv().reset(seed=0)
    a, _ = m.predict(obs, deterministic=True)
    assert a.shape == (3,) and np.all(np.abs(a) <= 1 + 1e-6)


def test_missing_model_raises():
    from training.evaluate_td3_adaptive_pid import load_model
    with pytest.raises(FileNotFoundError):
        load_model("no_such_adaptive_model")


def test_deterministic_evaluation(tiny_model_path):
    from training.evaluate_td3_adaptive_pid import load_model, run_scenario
    m = load_model(tiny_model_path)
    r1 = run_scenario(m, "vin", 10.0, t_dist=0.01, t_end=0.02)
    r2 = run_scenario(m, "vin", 10.0, t_dist=0.01, t_end=0.02)
    for k in ("vout", "il", "duty", "Kp", "Ki", "Kd"):
        assert np.array_equal(r1[k], r2[k])


def test_evaluation_finite_and_safe(tiny_model_path):
    from training.evaluate_td3_adaptive_pid import evaluate, load_model
    m = load_model(tiny_model_path)
    results, df = evaluate(m, t_dist=0.01, t_end=0.02)
    assert list(df.index) == list(config.PID_SCENARIOS) and df["all_finite"].all()
    need = {"rise_time_ms", "settling_time_ms", "overshoot_pct", "undershoot_V", "ss_error_V",
            "rmse_V", "max_dev_after_V", "recovery_time_ms", "duty_variation", "max_IL_A",
            "min_IL_A"}
    assert need <= set(df.columns)
    for r in results.values():
        assert r["duty"].min() >= config.DUTY_MIN - 1e-12
        assert r["duty"].max() <= config.DUTY_MAX + 1e-12
        assert np.all(np.isfinite(r["vout"])) and r["il_min"].min() >= -1e-9
        for g, (lo, hi) in zip(GAINS, RANGES):
            assert r[g].min() >= lo - 1e-15 and r[g].max() <= hi + 1e-15


def test_zero_policy_evaluation_equals_pid_metrics():
    """Evaluating the zero policy through the adaptive harness gives the PID metrics."""
    from training.evaluate_td3_adaptive_pid import run_scenario
    ref = run_pid_scenario("vin", 10.0)
    r = run_scenario(None, "vin", 10.0)
    assert np.array_equal(r["vout"], ref["vout"]) and np.array_equal(r["duty"], ref["duty"])


# ============================================================ clamping
def test_unsafe_gain_proposals_are_clamped():
    c = AdaptivePID()
    out = c.clamp_gains(dict(Kp=1e3, Ki=-5.0, Kd=float("nan")))
    assert out["Kp"] == c.ranges["Kp"][1]
    assert out["Ki"] == c.ranges["Ki"][0]
    assert out["Kd"] == BASE[2]                      # non-finite -> baseline
    for g in GAINS:
        assert c.ranges[g][0] <= out[g] <= c.ranges[g][1]


def test_propose_gains_never_applies_an_unsafe_set():
    c = AdaptivePID()
    for _ in range(600):
        c.propose_gains(Kp=1.0, Ki=1e6, Kd=1e-12)    # Kp, Ki far above / Kd far below the box
        for g in GAINS:
            lo, hi = c.ranges[g]
            assert lo - 1e-15 <= c.gains[g] <= hi + 1e-15
            assert c.pid.__dict__[g] == c.gains[g]   # what the PID really uses
    assert c.gains["Kp"] == pytest.approx(c.ranges["Kp"][1], rel=1e-6)
    assert c.gains["Ki"] == pytest.approx(c.ranges["Ki"][1], rel=1e-6)
    assert c.gains["Kd"] == pytest.approx(c.ranges["Kd"][0], rel=1e-6)


def test_propose_gains_is_rate_limited():
    c = AdaptivePID()
    prev = c.gain_array
    for _ in range(100):
        c.propose_gains(Kp=1e3, Ki=1e-9, Kd=1e3)
        now = c.gain_array
        assert np.all(np.abs(np.log(now / prev)) <= config.ADAPT_RATE_LIMIT + 1e-12)
        prev = now


def test_gain_in_the_pid_always_matches_the_bounded_value():
    c = AdaptivePID()
    rng = np.random.default_rng(5)
    for _ in range(3000):
        c.adapt(rng.uniform(-9, 9, size=3))
        assert (c.pid.Kp, c.pid.Ki, c.pid.Kd) == tuple(c.gains[g] for g in GAINS)


# ====================================================== baseline recovery
DT = config.DT
STEPS = lambda sec: int(round(sec / DT))


def feed(c, err, n, il=2.4, ref=24.0):
    """Run the supervisor for n PID steps with a constant tracking error."""
    for _ in range(n):
        c.update(ref, ref - err, DT, il)


def test_recovery_engages_on_persistent_large_error():
    c = AdaptivePID()
    feed(c, 6.0, STEPS(config.ADAPT_REC_PERSIST * 0.4))
    assert not c.recovering                          # brief error: no engage
    feed(c, 6.0, STEPS(config.ADAPT_REC_PERSIST + 4 * config.ADAPT_REC_ERR_TAU))
    assert c.recovering and c.recovery_events == 1


def test_short_error_spike_does_not_engage_recovery():
    c = AdaptivePID()
    for _ in range(5):
        feed(c, 8.0, STEPS(1e-3)); feed(c, 0.0, STEPS(10e-3))
    assert not c.recovering


def test_recovery_engages_immediately_on_overcurrent_or_overvoltage():
    c = AdaptivePID()
    c.update(24.0, 24.0, DT, config.ADAPT_REC_IL_WARN + 1.0)
    assert c.recovering
    c2 = AdaptivePID()
    c2.update(24.0, 24.0 * config.ADAPT_REC_V_WARN_RATIO + 0.5, DT, 2.0)
    assert c2.recovering


def test_nominal_inrush_does_not_trigger_the_current_warning():
    """The baseline PID/SMC cold-start inrush (~9 A) must stay below the warning level."""
    assert config.ADAPT_REC_IL_WARN > 9.5


def test_recovery_returns_gains_to_baseline_smoothly():
    c = AdaptivePID()
    settle(c, np.array([1.0, -1.0, 1.0]), n=300)       # drive gains to the box corner
    corner = c.gain_array.copy()
    assert not np.allclose(corner, BASE)
    feed(c, 6.0, STEPS(0.05))                          # persistent error -> recovery engages
    assert c.recovering
    prev = c.gain_array
    for _ in range(300):
        c.adapt(np.array([1.0, -1.0, 1.0]))            # the agent keeps asking for the corner
        now = c.gain_array
        assert np.all(np.abs(np.log(now / prev)) <= config.ADAPT_RATE_LIMIT + 1e-12)  # smooth
        prev = now
    assert np.allclose(c.gain_array, BASE, rtol=1e-3)  # ...and is ignored while recovering


def test_recovery_releases_after_the_error_clears():
    c = AdaptivePID()
    feed(c, 6.0, STEPS(0.05))
    assert c.recovering
    feed(c, 0.0, STEPS(config.ADAPT_REC_MIN_HOLD * 0.3))
    assert c.recovering                                # minimum hold time
    feed(c, 0.0, STEPS(0.05))
    assert not c.recovering
    c.adapt(np.array([1.0, 0.0, 0.0]))                 # the agent regains control
    assert c.gains["Kp"] > BASE[0]


def test_recovery_can_be_disabled():
    c = AdaptivePID(recovery=False)
    feed(c, 20.0, STEPS(0.1), il=50.0)
    assert not c.recovering
    c.adapt(np.array([1.0, 0, 0]))
    assert c.gains["Kp"] > BASE[0]


def test_recovery_leaves_the_zero_action_baseline_untouched():
    c = AdaptivePID()
    feed(c, 6.0, STEPS(0.05))
    settle(c, np.zeros(3), n=50)
    assert c.recovering and [c.gains[g] for g in GAINS] == list(BASE)


def test_duty_stays_in_limits_while_recovering_with_extreme_inputs():
    c = AdaptivePID()
    rng = np.random.default_rng(6)
    for k in range(6000):
        if k % 10 == 0:
            c.adapt(rng.uniform(-3, 3, size=3))
        d = c.update(rng.uniform(0, 60), rng.uniform(-5, 90), DT, rng.uniform(0, 40))
        assert config.DUTY_MIN <= d <= config.DUTY_MAX and math.isfinite(d)


def test_env_reports_and_observes_recovery():
    e = AdaptivePIDEnv(disturbance_prob=0.0)
    e.reset(seed=0, options=dict(vin=10.0, load=30.0, vref=28.0, init="cold", disturbance=None))
    seen = False
    for _ in range(400):
        obs, r, term, trunc, info = e.step(np.array([1.0, 1.0, -1.0], np.float32))
        assert obs[9] == float(info["recovering"])
        seen |= info["recovering"]
        assert np.all(np.isfinite(obs)) and np.isfinite(r)
        if term or trunc:
            break
    assert seen                                         # cold start to 28 V: supervisor engages


def test_gains_return_to_baseline_inside_the_env_when_error_persists():
    e = AdaptivePIDEnv(disturbance_prob=0.0)
    e.reset(seed=0, options=dict(vin=12.0, load=20.0, vref=24.0, init="cold", disturbance=None))
    for _ in range(1200):
        _, _, term, trunc, info = e.step(np.array([1.0, -1.0, 1.0], np.float32))
        if term or trunc:
            break
    assert not (term and info["failure"])
