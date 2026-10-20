"""Deterministic evaluation of a trained TD3 controller on the five scenarios
used for the PID and SMC (cold start -> disturbance at 150 ms -> 300 ms).

    python training/evaluate_td3.py --name td3_boost

Measurement notes
* The agent acts every 200 us (4 converter steps); the duty is held for 4 converter steps.
  Time series are recorded at EVERY 50 us converter step (info["trace"]), the same sampling
  as the PID, SMC and adaptive controllers, so all metrics are directly comparable.
  (Earlier direct-TD3 tables were sampled at 200 us; the duty-variation scaling is unchanged.)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from analysis import plots
from environment.boost_env import BoostConverterEnv
from experiments.run_pid import compute_metrics


def load_model(name=config.TD3_MODEL_NAME):
    from stable_baselines3 import TD3
    path = Path(name) if str(name).endswith(".zip") else config.MODELS_DIR / f"{name}.zip"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - train first (training/train_td3.py)")
    return TD3.load(path, device="cpu")


def run_scenario(model, kind, value, t_dist=config.SCENARIO_T_DIST,
                 t_end=config.SCENARIO_T_END, seed=0):
    """Run one deterministic episode. Returns a dict like experiments.run_pid.run_scenario."""
    env = BoostConverterEnv(max_steps=int(round(t_end / (config.DT * config.ENV_CONTROL_STEPS))),
                            disturbance_prob=0.0)
    env.record_trace = True
    step_d = int(round(t_dist / env.dt))
    dist = None if kind is None else dict(step=step_d, vin=value if kind == "vin" else None,
                                          load=value if kind == "load" else None)
    obs, info = env.reset(seed=seed, options=dict(vin=config.VIN, load=config.R, vref=config.VREF,
                                                  init="cold", disturbance=dist))
    cols = {k: [] for k in ("t", "vout", "il", "duty", "vref", "il_max", "il_min", "reward")}
    finite, failure, terminated = True, None, False
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, terminated, truncated, info = env.step(action)
        finite &= bool(np.all(np.isfinite(obs)) and np.isfinite(r))
        tr = info["trace"]
        for k in ("t", "vout", "il", "duty"):
            cols[k].extend(tr[k])
        cols["vref"].extend([info["Vref"]] * len(tr["t"]))
        cols["il_max"].append(info["IL_max"]); cols["il_min"].append(info["IL_min"])
        cols["reward"].append(r)
        if terminated or truncated:
            failure = info["failure"]
            break
    out = {k: np.asarray(v) for k, v in cols.items()}
    out.update(disturbed=kind is not None, finite=finite, terminated=terminated, failure=failure)
    return out


def evaluate(model, t_dist=config.SCENARIO_T_DIST, t_end=config.SCENARIO_T_END, save_prefix=None):
    results, rows = {}, []
    for name, (kind, val, label) in config.PID_SCENARIOS.items():
        r = run_scenario(model, kind, val, t_dist=t_dist, t_end=t_end)
        r["label"] = label
        results[name] = r
        m = compute_metrics(r, t_dist=t_dist)
        m["max_IL_A"], m["min_IL_A"] = float(r["il_max"].max()), float(r["il_min"].min())
        m["episode_return"] = float(r["reward"].sum())
        m["terminated_early"] = bool(r["terminated"])
        m["failure"] = r["failure"] or ""
        m["all_finite"] = bool(r["finite"])
        rows.append(dict(scenario=name, **m))
        if save_prefix:
            pd.DataFrame({k: r[k] for k in ("t", "vout", "il", "duty", "vref")}).to_csv(
                config.DATA_DIR / f"{save_prefix}_{name}.csv", index=False)
    df = pd.DataFrame(rows).set_index("scenario")
    if save_prefix:
        df.to_csv(config.RESULTS_DIR / f"{save_prefix}_metrics.csv")
        td = t_dist
        plots.scenario_grid(results, "vout", "TD3: Vref vs Vout", f"{save_prefix}_vref_vs_vout.png",
                            t_dist=td, show_ref=True)
        plots.scenario_grid(results, "error", "TD3: voltage error", f"{save_prefix}_error.png",
                            t_dist=td, ylim=(-6, 6))
        plots.scenario_grid(results, "duty", "TD3: duty cycle", f"{save_prefix}_duty.png", t_dist=td)
        plots.scenario_grid(results, "il", "TD3: inductor current",
                            f"{save_prefix}_inductor_current.png", t_dist=td)
    return results, df


def print_with_baselines(df, label):
    cols = ["settling_time_ms", "overshoot_pct", "undershoot_V", "ss_error_final_V", "rmse_V",
            "max_dev_after_V", "recovery_time_ms", "duty_variation", "max_IL_A", "min_IL_A"]
    pd.set_option("display.width", 250, "display.max_columns", 30,
                  "display.float_format", "{:.4g}".format)
    print(f"\n{label} metrics (rows = metrics, columns = scenarios)")
    print(df[cols].T)
    extra = df[["terminated_early", "all_finite"]]
    print("\nterminated early / all finite:\n", extra.T)
    for pre in ("pid", "smc"):
        f = config.RESULTS_DIR / f"{pre}_metrics.csv"
        if f.exists():
            b = pd.read_csv(f, index_col=0)
            print(f"\n{pre.upper()} baseline (same scenarios, from {f.name})")
            print(b[[c for c in cols if c in b.columns]].T)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--name", default=config.TD3_MODEL_NAME)
    a = p.parse_args(argv)
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    model = load_model(a.name)
    prefix = "td3" if a.name == config.TD3_MODEL_NAME else a.name   # smoke runs never clobber td3_*
    _, df = evaluate(model, save_prefix=prefix)
    print_with_baselines(df, f"TD3 ({a.name})")
    print("\nFigures/data saved under", config.RESULTS_DIR)
    return df


if __name__ == "__main__":
    main()
