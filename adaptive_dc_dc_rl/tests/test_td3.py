import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from environment.boost_env import BoostConverterEnv, action_to_duty, duty_to_action


# ---------------------------------------------------------- action mapping
def test_action_mapping_endpoints_and_midpoint():
    assert action_to_duty(-1.0) == pytest.approx(config.DUTY_MIN)
    assert action_to_duty(0.0) == pytest.approx(0.5)
    assert action_to_duty(1.0) == pytest.approx(config.DUTY_MAX)


def test_action_mapping_clips_out_of_range():
    assert action_to_duty(-7.0) == pytest.approx(config.DUTY_MIN)
    assert action_to_duty(7.0) == pytest.approx(config.DUTY_MAX)


def test_action_mapping_is_monotonic_and_round_trips():
    a = np.linspace(-1, 1, 101)
    d = np.array([action_to_duty(x) for x in a])
    assert np.all(np.diff(d) > 0)
    back = np.array([duty_to_action(x)[0] for x in d])
    assert np.allclose(back, a, atol=1e-6)
    assert duty_to_action(0.5).shape == (1,) and duty_to_action(0.5).dtype == np.float32


def test_env_uses_the_same_mapping():
    env = BoostConverterEnv(disturbance_prob=0.0)
    env.reset(seed=0)
    for a in (-1.0, -0.3, 0.0, 0.7, 1.0):
        env.step(np.array([a], np.float32))
        assert env.converter.duty == pytest.approx(action_to_duty(a))
    assert env.action_space.low[0] == -1 and env.action_space.high[0] == 1


# ------------------------------------------------- tiny model: save / load
@pytest.fixture(scope="module")
def tiny_model_path(tmp_path_factory):
    """Very short real TD3 training run (a few hundred steps) saved to a temp dir."""
    from training import train_td3
    out = tmp_path_factory.mktemp("models")
    old = config.MODELS_DIR
    config.MODELS_DIR = out
    try:
        path = train_td3.train(total_timesteps=300, seed=0, model_name="tiny",
                               learning_starts=100, tensorboard=False)
    finally:
        config.MODELS_DIR = old
    return Path(path)


def test_model_saved_with_info_file(tiny_model_path):
    assert tiny_model_path.exists() and tiny_model_path.suffix == ".zip"
    assert tiny_model_path.with_name("tiny_info.json").exists()


def test_model_loads_and_predicts_valid_actions(tiny_model_path):
    from training.evaluate_td3 import load_model
    model = load_model(tiny_model_path)
    obs, _ = BoostConverterEnv().reset(seed=0)
    action, _ = model.predict(obs, deterministic=True)
    assert action.shape == (1,) and np.all(np.abs(action) <= 1.0 + 1e-6)
    assert model.action_space.low[0] == -1 and model.action_space.high[0] == 1


def test_load_missing_model_raises():
    from training.evaluate_td3 import load_model
    with pytest.raises(FileNotFoundError):
        load_model("definitely_not_a_model")


# ---------------------------------------------------- deterministic evaluation
def test_deterministic_evaluation_is_repeatable(tiny_model_path):
    from training.evaluate_td3 import load_model, run_scenario
    model = load_model(tiny_model_path)
    r1 = run_scenario(model, "vin", 10.0, t_dist=0.01, t_end=0.02)
    r2 = run_scenario(model, "vin", 10.0, t_dist=0.01, t_end=0.02)
    for k in ("vout", "il", "duty", "reward"):
        assert np.array_equal(r1[k], r2[k])


def test_evaluation_finite_and_duty_in_range(tiny_model_path):
    from training.evaluate_td3 import load_model, run_scenario
    model = load_model(tiny_model_path)
    for kind, val in ((None, None), ("vin", 14.0), ("load", 10.0)):
        r = run_scenario(model, kind, val, t_dist=0.01, t_end=0.02)
        assert r["finite"]
        for k in ("vout", "il", "duty", "reward"):
            assert np.all(np.isfinite(r[k]))
        assert r["duty"].min() >= config.DUTY_MIN - 1e-12
        assert r["duty"].max() <= config.DUTY_MAX + 1e-12
        assert r["il_min"].min() >= -1e-9


def test_evaluation_metrics_table_has_required_columns(tiny_model_path):
    from training.evaluate_td3 import evaluate, load_model
    model = load_model(tiny_model_path)
    _, df = evaluate(model, t_dist=0.01, t_end=0.02)
    need = {"rise_time_ms", "settling_time_ms", "overshoot_pct", "undershoot_V",
            "ss_error_V", "rmse_V", "max_dev_after_V", "recovery_time_ms",
            "duty_variation", "max_IL_A", "min_IL_A"}
    assert need <= set(df.columns)
    assert list(df.index) == list(config.PID_SCENARIOS)
    assert df["all_finite"].all()


def test_untrained_random_policy_never_leaves_duty_range():
    env = BoostConverterEnv()
    rng = np.random.default_rng(0)
    env.reset(seed=0)
    for _ in range(300):
        _, r, term, trunc, info = env.step(rng.uniform(-3, 3, size=1))
        assert np.isfinite(r)
        assert config.DUTY_MIN <= env.converter.duty <= config.DUTY_MAX
        if term or trunc:
            env.reset()
