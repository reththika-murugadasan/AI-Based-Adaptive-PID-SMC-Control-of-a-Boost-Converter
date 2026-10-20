import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from environment.boost_env import BoostConverterEnv
from experiments import final_evaluation as fe
from experiments import train_final

EVAL_SCENARIOS = [(12.0, 20.0, 24.0)]


# ============================================== configurable training randomisation
def sample(env, n=400, **opt):
    out = []
    for s in range(n):
        env.reset(seed=s, options=opt or None)
        out.append((env.converter.Vin, env.converter.R, env.converter.Vref, env._disturbance))
    return out


def test_initial_condition_ranges_are_configurable_and_respected():
    env = BoostConverterEnv(vin_range=(11.0, 13.0), r_range=(15.0, 25.0), vref_range=(22.0, 26.0))
    s = sample(env)
    for i, (lo, hi) in enumerate(((11, 13), (15, 25), (22, 26))):
        vals = [x[i] for x in s]
        assert min(vals) >= lo and max(vals) <= hi and max(vals) - min(vals) > 0.8 * (hi - lo)


def test_disturbance_probability_extremes():
    assert all(d is None for *_, d in sample(BoostConverterEnv(disturbance_prob=0.0), n=100))
    assert all(d is not None for *_, d in sample(BoostConverterEnv(disturbance_prob=1.0), n=100))


def test_disturbance_time_follows_the_configured_window():
    env = BoostConverterEnv(disturbance_prob=1.0, disturbance_window=(0.4, 0.5), max_steps=400)
    steps = [d["step"] for *_, d in sample(env, n=300)]
    assert min(steps) >= 160 and max(steps) < 200 and len(set(steps)) > 10


def test_disturbance_magnitudes_follow_the_configured_ranges():
    env = BoostConverterEnv(disturbance_prob=1.0, disturbance_kinds=("vin",),
                            disturbance_vin_delta=(1.5, 2.5), vin_range=(8.0, 20.0))
    for vin, R, vref, d in sample(env, n=300):
        assert d["load"] is None and d["vref"] is None
        assert 1.5 - 1e-9 <= abs(d["vin"] - vin) <= 2.5 + 1e-9


def test_load_disturbance_ratio_and_clipping():
    env = BoostConverterEnv(disturbance_prob=1.0, disturbance_kinds=("load",),
                            disturbance_load_ratio=(0.5, 2.0))
    for vin, R, vref, d in sample(env, n=300):
        assert d["vin"] is None
        assert config.ENV_R_RANGE[0] <= d["load"] <= config.ENV_R_RANGE[1]
        lo, hi = max(0.5 * R, config.ENV_R_RANGE[0]), min(2.0 * R, config.ENV_R_RANGE[1])
        assert lo - 1e-9 <= d["load"] <= hi + 1e-9


def test_vref_disturbances_are_available_but_off_by_default():
    assert "vref" not in config.ENV_DISTURBANCE_KINDS
    env = BoostConverterEnv(disturbance_prob=1.0, disturbance_kinds=("vref",),
                            disturbance_vref_delta=(1.0, 3.0))
    for vin, R, vref, d in sample(env, n=100):
        assert d["vref"] is not None and config.ENV_VREF_RANGE[0] <= d["vref"] <= config.ENV_VREF_RANGE[1]


def test_invalid_disturbance_configuration_rejected():
    with pytest.raises(ValueError):
        BoostConverterEnv(disturbance_kinds=("bogus",))
    with pytest.raises(ValueError):
        BoostConverterEnv(disturbance_window=(0.8, 0.2))


def test_applied_disturbance_changes_the_converter_at_the_chosen_step():
    env = BoostConverterEnv(disturbance_prob=1.0, disturbance_kinds=("both",))
    env.reset(seed=3)
    d = env._disturbance
    vin0, r0 = env.converter.Vin, env.converter.R
    for _ in range(d["step"]):                                  # before the chosen step: unchanged
        env.step(np.array([0.0], np.float32))
        assert (env.converter.Vin, env.converter.R) == (vin0, r0)
    env.step(np.array([0.0], np.float32))                       # the disturbance step itself
    assert env.converter.Vin == pytest.approx(d["vin"]) and env.converter.R == pytest.approx(d["load"])


def test_same_seed_gives_the_same_randomised_episode_and_different_seeds_differ():
    a, b = BoostConverterEnv(), BoostConverterEnv()
    for s in range(20):
        a.reset(seed=s); b.reset(seed=s)
        assert (a.converter.Vin, a.converter.R, a.converter.Vref, a._disturbance) == \
               (b.converter.Vin, b.converter.R, b.converter.Vref, b._disturbance)
    assert len({round(x[0], 6) for x in sample(BoostConverterEnv(), n=50)}) > 40


def test_training_distribution_never_samples_the_fixed_evaluation_points():
    """Evaluation uses exactly (12, 20, 24); the continuous training sampler never hits it."""
    for vin, R, vref, d in sample(BoostConverterEnv(), n=1000):
        assert (vin, R, vref) not in EVAL_SCENARIOS


def test_randomisation_covers_the_evaluation_conditions():
    s = sample(BoostConverterEnv(), n=500)
    assert min(x[0] for x in s) < 10.5 and max(x[0] for x in s) > 13.5      # Vin 10 .. 14
    assert min(x[1] for x in s) < 12 and max(x[1] for x in s) > 28           # R 10 .. 30
    kinds = {d["kind"] for *_, d in s if d is not None}
    assert kinds == set(config.ENV_DISTURBANCE_KINDS)


def test_safety_limits_are_unchanged():
    assert (config.DUTY_MIN, config.DUTY_MAX) == (0.05, 0.95)
    assert (config.ENV_V_MAX, config.ENV_I_MAX) == (60.0, 30.0)
    assert config.ENV_REWARD["fail_penalty"] == 50.0


# ===================================================== protection-trip detection
def make_series(v=24.0, il=2.4, **kw):
    n = 50
    s = dict(t=np.arange(n) * 5e-5, vout=np.full(n, v), il=np.full(n, il))
    s.update(kw)
    return s


def test_protection_trip_detection():
    assert fe.protection_trip(make_series()) == (False, "")
    assert fe.protection_trip(make_series(v=config.ENV_V_MAX + 1))[0]
    assert fe.protection_trip(make_series(il=config.ENV_I_MAX + 1))[0]
    assert fe.protection_trip(make_series(il=-0.5))[0]
    assert fe.protection_trip(make_series(v=float("nan")))[0]
    assert fe.protection_trip(make_series(terminated=True, failure="over-voltage trip")) == (
        True, "over-voltage trip")
    assert not fe.protection_trip(make_series(terminated=False, failure=None))[0]


# ============================================================== statistics
def test_summarise_mean_std_min_max_and_undefined_counts():
    rows = []
    for seed, (rec, ov) in enumerate([(10.0, 1.0), (20.0, 2.0), (np.nan, 3.0)]):
        row = dict(controller="c", scenario="s", seed=seed)
        row.update({m: 0.0 for m in fe.METRICS})
        row.update(recovery_time_ms=rec, overshoot_pct=ov)
        rows.append(row)
    s = fe.summarise(pd.DataFrame(rows))
    r = s[(s.metric == "recovery_time_ms")].iloc[0]
    assert (r.n, r.n_undefined) == (3, 1) and r["mean"] == 15.0
    assert r["std"] == pytest.approx(np.std([10, 20], ddof=1)) and (r["min"], r["max"]) == (10, 20)
    o = s[(s.metric == "overshoot_pct")].iloc[0]
    assert o["mean"] == 2.0 and o["std"] == pytest.approx(1.0)


def test_summarise_single_run_has_zero_std():
    row = dict(controller="c", scenario="s", seed="")
    row.update({m: 1.5 for m in fe.METRICS})
    r = fe.summarise(pd.DataFrame([row])).iloc[0]
    assert r.n == 1 and r["std"] == 0.0


# ================================================ reproducible training (seeded)
def _train(tmp, controller, seed, steps=240, name=None):
    old = config.MODELS_DIR
    config.MODELS_DIR = tmp
    try:
        mod = {"direct": "training.train_td3", "adaptive-pid": "training.train_td3_adaptive_pid",
               "adaptive-smc": "training.train_td3_adaptive_smc"}[controller]
        import importlib
        m = importlib.import_module(mod)
        return Path(m.train(steps, seed, name or f"{controller}_{seed}", learning_starts=80,
                            tensorboard=False))
    finally:
        config.MODELS_DIR = old


def _actor_params(path):
    from stable_baselines3 import TD3
    m = TD3.load(path, device="cpu")
    return np.concatenate([p.detach().numpy().ravel() for p in m.actor.parameters()])


@pytest.mark.parametrize("controller", ["direct", "adaptive-pid"])
def test_training_is_reproducible_for_a_fixed_seed(tmp_path, controller):
    a = _actor_params(_train(tmp_path, controller, 0, name="a"))
    b = _actor_params(_train(tmp_path, controller, 0, name="b"))
    c = _actor_params(_train(tmp_path, controller, 1, name="c"))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_model_naming_and_training_info_records_the_randomisation(tmp_path):
    assert train_final.model_name("adaptive-smc", "final", 2) == "td3_adaptive_smc_final_seed2"
    path = _train(tmp_path, "direct", 0, name="meta")
    info = json.loads(path.with_name("meta_info.json").read_text())
    assert info["seed"] == 0
    assert info["hyperparameters"]["ENV_VIN_RANGE"] == list(config.ENV_VIN_RANGE)
    assert "ENV_DISTURBANCE_VIN_DELTA" in info["hyperparameters"]


def test_train_final_cli_validates_arguments():
    with pytest.raises(SystemExit):
        train_final.main(["--controllers", "bogus"])


# ================================================== whole pipeline on tiny models
@pytest.fixture(scope="module")
def tiny_models(tmp_path_factory):
    d = tmp_path_factory.mktemp("final_models")
    for c in ("direct", "adaptive-pid", "adaptive-smc"):
        base = _train(d, c, 0, name=train_final.model_name(c, "tiny", 0))
        for seed in (1, 2):           # reuse the seed-0 model: tests the plumbing, not the learning
            (d / (train_final.model_name(c, "tiny", seed) + ".zip")).write_bytes(base.read_bytes())
    return d


@pytest.fixture(scope="module")
def pipeline(tiny_models, tmp_path_factory):
    out = tmp_path_factory.mktemp("final_out")
    res = fe.run(tag="tiny", seeds=(0, 1, 2), models_dir=tiny_models, out_dir=out,
                 t_dist=0.01, t_end=0.02, plots=True)
    return res


def test_combined_csv_has_all_controllers_seeds_and_scenarios(pipeline):
    df, summary, store, out, figs = pipeline
    csv = pd.read_csv(out / "final_metrics_by_run.csv")
    assert set(csv.controller) == set(fe.CONTROLLERS)
    assert len(csv) == (2 * 1 + 3 * 3) * 5                      # 2 fixed + 3 RL x 3 seeds, 5 scenarios
    for ctrl, (_, rl) in fe.CONTROLLERS.items():
        g = csv[csv.controller == ctrl]
        assert set(g.scenario) == set(config.PID_SCENARIOS)
        assert g.seed.dropna().nunique() == (3 if rl else 0)
    need = {"rise_time_ms", "settling_time_ms", "overshoot_pct", "undershoot_V", "ss_error_V",
            "ss_error_final_V", "rmse_V", "max_dev_after_V", "recovery_time_ms", "duty_variation",
            "max_IL_A", "min_IL_A", "protection_trip"}
    assert need <= set(csv.columns)


def test_all_controllers_use_identical_scenarios_and_time_grid(pipeline):
    store = pipeline[2]
    for sc in config.PID_SCENARIOS:
        grids = [s["t"] for c in store for s in store[c][sc] if len(s["t"]) == len(store["fixed_pid"][sc][0]["t"])]
        assert len(grids) == 11 and all(np.allclose(g, grids[0]) for g in grids)   # 50 us everywhere
        refs = {round(float(s["vref"][0]), 9) for c in store for s in store[c][sc]}
        assert refs == {config.VREF}


def test_no_rankings_or_scores_are_produced(pipeline):
    df, summary, store, out, figs = pipeline
    banned = ("rank", "score", "winner", "best")
    assert not any(b in c.lower() for c in df.columns for b in banned)
    assert not any(b in c.lower() for c in summary.columns for b in banned)
    text = (out / "final_report.md").read_text().lower()
    assert "no ranking" in text and "winner:" not in text and "rank 1" not in text


def test_statistics_over_seeds_are_reported_for_rl_only(pipeline):
    summary = pipeline[1]
    rl = summary[summary.controller == "adaptive_pid"]
    fixed = summary[summary.controller == "fixed_pid"]
    assert set(rl.n) == {3} and set(fixed.n) == {1}
    assert {"mean", "std", "min", "max", "n_undefined"} <= set(summary.columns)


def test_report_files_figures_and_metadata_exist(pipeline, tiny_models):
    df, summary, store, out, figs = pipeline
    for f in ("final_metrics_by_run.csv", "final_summary.csv", "final_report.md", "run_metadata.json"):
        assert (out / f).exists()
    names = {p.name for p in figs}
    assert {"final_vout_tracking.png", "final_voltage_error.png", "final_duty_cycle.png",
            "final_inductor_current.png", "final_adaptive_pid_gains.png",
            "final_adaptive_smc_params.png", "final_disturbance_recovery.png"} <= names
    assert all(p.stat().st_size > 5000 for p in figs)
    meta = json.loads((out / "run_metadata.json").read_text())
    assert meta["seeds"] == [0, 1, 2] and len(meta["models"]) == 9
    assert all(len(v["sha256"]) == 64 for v in meta["models"].values())


def test_report_has_protection_trip_table(pipeline):
    text = (pipeline[3] / "final_report.md").read_text()
    assert "protection trips" in text and "supervisor engagements" in text


def test_rl_parameters_stay_inside_their_bounds_in_the_final_runs(pipeline):
    store = pipeline[2]
    for s in (r for runs in store["adaptive_pid"].values() for r in runs):
        for g, (lo, hi) in zip(("Kp", "Ki", "Kd"), (config.ADAPT_KP_RANGE, config.ADAPT_KI_RANGE,
                                                    config.ADAPT_KD_RANGE)):
            assert s[g].min() >= lo - 1e-15 and s[g].max() <= hi + 1e-15
    from controllers.adaptive_smc import default_ranges
    for s in (r for runs in store["adaptive_smc"].values() for r in runs):
        for p, (lo, hi) in default_ranges().items():
            assert s[p].min() >= lo - 1e-15 and s[p].max() <= hi + 1e-15


def test_fixed_baselines_in_the_final_pipeline_match_the_stage_results(pipeline):
    """The fixed PID/SMC rows are computed by the same code as the stage CSVs (full-length check
    is done in test_fixed_baselines_full_length); here: identical series for a short run."""
    from controllers.pid import PIDController
    from experiments.run_pid import run_scenario
    ref = run_scenario("vin", 10.0, controller=PIDController(), t_dist=0.01, t_end=0.02)
    got = pipeline[2]["fixed_pid"]["vin_12_to_10"][0]
    assert np.array_equal(ref["vout"], got["vout"])


def test_missing_models_give_a_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="train_final.py"):
        fe.run(tag="nope", seeds=(0,), models_dir=tmp_path, out_dir=tmp_path / "o")


def test_direct_td3_series_use_50us_sampling(tiny_models):
    from training.evaluate_td3 import load_model, run_scenario
    m = load_model(fe.model_path("direct", "tiny", 0, tiny_models))
    r = run_scenario(m, None, None, t_dist=0.01, t_end=0.02)
    if not r["terminated"]:
        assert len(r["t"]) == int(round(0.02 / config.DT))
    assert np.allclose(np.diff(r["t"]), config.DT)
