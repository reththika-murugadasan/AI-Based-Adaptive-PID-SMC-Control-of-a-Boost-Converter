"""Deterministic evaluation of the TD3 adaptive PID on the five standard scenarios.

    python training/evaluate_td3_adaptive_pid.py --name td3_adaptive_pid_smoke

The agent acts every 200 us; the PID inside runs at 50 us.  Time series are
recorded at EVERY 50 us converter step (info["trace"]), so the metrics use the same
sampling as the PID and SMC baselines.  ``model=None`` evaluates the zero-action
policy, i.e. the baseline PID itself.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from analysis import plots
from environment.adaptive_pid_env import AdaptivePIDEnv
from experiments.run_pid import compute_metrics


def load_model(name=config.ADAPT_MODEL_NAME):
    from stable_baselines3 import TD3
    path = Path(name) if str(name).endswith(".zip") else config.MODELS_DIR / f"{name}.zip"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - train first "
                                f"(training/train_td3_adaptive_pid.py)")
    return TD3.load(path, device="cpu")


def run_scenario(model, kind, value, t_dist=config.SCENARIO_T_DIST,
                 t_end=config.SCENARIO_T_END, seed=0):
    """One deterministic episode; returns arrays at 50 us resolution."""
    env = AdaptivePIDEnv(max_steps=int(round(t_end / (config.DT * config.ENV_CONTROL_STEPS))),
                         disturbance_prob=0.0)
    env.record_trace = True
    step_d = int(round(t_dist / env.dt))
    dist = None if kind is None else dict(step=step_d, vin=value if kind == "vin" else None,
                                          load=value if kind == "load" else None)
    obs, info = env.reset(seed=seed, options=dict(vin=config.VIN, load=config.R, vref=config.VREF,
                                                  init="cold", disturbance=dist))
    keys = ("t", "vout", "il", "duty", "il_max", "il_min", "Kp", "Ki", "Kd", "recovering")
    cols = {k: [] for k in keys}
    reward, finite, failure, terminated, n_events = 0.0, True, None, False, 0
    n_sub = env.control_steps
    while True:
        if model is None:
            action = np.zeros(3, np.float32)
        else:
            action, _ = model.predict(obs, deterministic=True)
        obs, r, terminated, truncated, info = env.step(action)
        reward += r
        finite &= bool(np.all(np.isfinite(obs)) and np.isfinite(r))
        tr = info["trace"]
        for k in ("t", "vout", "il", "duty"):
            cols[k].extend(tr[k])
        cols["il_max"].append(info["IL_max"]); cols["il_min"].append(info["IL_min"])
        for g in ("Kp", "Ki", "Kd"):
            cols[g].extend([info[g]] * len(tr["t"]))
        cols["recovering"].extend([float(info["recovering"])] * len(tr["t"]))
        n_events = info["recovery_events"]
        if terminated or truncated:
            failure = info["failure"]
            break
    out = {k: np.asarray(v) for k, v in cols.items()}
    out["vref"] = np.full_like(out["vout"], env.converter.Vref)
    out.update(disturbed=kind is not None, finite=finite, terminated=terminated, recovery_events=n_events,
               failure=failure, episode_return=reward)
    return out


def evaluate(model, t_dist=config.SCENARIO_T_DIST, t_end=config.SCENARIO_T_END,
             save_prefix=None):
    results, rows = {}, []
    for name, (kind, val, label) in config.PID_SCENARIOS.items():
        r = run_scenario(model, kind, val, t_dist=t_dist, t_end=t_end)
        r["label"] = label
        results[name] = r
        m = compute_metrics(r, t_dist=t_dist)
        m["max_IL_A"], m["min_IL_A"] = float(r["il_max"].max()), float(r["il_min"].min())
        m["Kp_min"], m["Kp_max"] = float(r["Kp"].min()), float(r["Kp"].max())
        m["Ki_min"], m["Ki_max"] = float(r["Ki"].min()), float(r["Ki"].max())
        m["Kd_min"], m["Kd_max"] = float(r["Kd"].min()), float(r["Kd"].max())
        m["recovery_events"] = int(r["recovery_events"])
        m["recovering_fraction"] = float(r["recovering"].mean())
        m["episode_return"] = float(r["episode_return"])
        m["terminated_early"] = bool(r["terminated"])
        m["failure"] = r["failure"] or ""
        m["all_finite"] = bool(r["finite"])
        rows.append(dict(scenario=name, **m))
        if save_prefix:
            pd.DataFrame({k: r[k] for k in ("t", "vout", "il", "duty", "vref", "Kp", "Ki", "Kd", "recovering")}
                         ).to_csv(config.DATA_DIR / f"{save_prefix}_{name}.csv", index=False)
    df = pd.DataFrame(rows).set_index("scenario")
    if save_prefix:
        config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(config.RESULTS_DIR / f"{save_prefix}_metrics.csv")
        td = t_dist
        lab = "Adaptive PID"
        plots.scenario_grid(results, "vout", f"{lab}: Vref vs Vout",
                            f"{save_prefix}_vref_vs_vout.png", t_dist=td, show_ref=True)
        plots.scenario_grid(results, "error", f"{lab}: voltage error", f"{save_prefix}_error.png",
                            t_dist=td, ylim=(-6, 6))
        plots.scenario_grid(results, "duty", f"{lab}: duty cycle", f"{save_prefix}_duty.png",
                            t_dist=td)
        plots.scenario_grid(results, "il", f"{lab}: inductor current",
                            f"{save_prefix}_inductor_current.png", t_dist=td)
        plots.gain_grid(results, f"{save_prefix}_gains.png",
                        base=dict(Kp=config.PID_KP, Ki=config.PID_KI, Kd=config.PID_KD),
                        bounds=dict(Kp=config.ADAPT_KP_RANGE, Ki=config.ADAPT_KI_RANGE,
                                    Kd=config.ADAPT_KD_RANGE), t_dist=td)
    return results, df


def print_comparison(df, label, prefix=None):
    cols = ["settling_time_ms", "overshoot_pct", "undershoot_V", "ss_error_final_V", "rmse_V",
            "max_dev_after_V", "recovery_time_ms", "duty_variation", "max_IL_A", "min_IL_A"]
    pd.set_option("display.width", 250, "display.max_columns", 30,
                  "display.float_format", "{:.4g}".format)
    print(f"\n{label} (rows = metrics, columns = scenarios)")
    print(df[cols].T)
    print("\nsupervisor activity:")
    print(df[["recovery_events", "recovering_fraction"]].T)
    print("\nactual gain excursion during the runs (min .. max) vs allowed range:")
    for g, rng in zip(("Kp", "Ki", "Kd"), (config.ADAPT_KP_RANGE, config.ADAPT_KI_RANGE,
                                           config.ADAPT_KD_RANGE)):
        base = {"Kp": config.PID_KP, "Ki": config.PID_KI, "Kd": config.PID_KD}[g]
        print(f"  {g}: used {df[g + '_min'].min():.5g} .. {df[g + '_max'].max():.5g} | "
              f"allowed [{rng[0]:.5g}, {rng[1]:.5g}] = baseline x [{rng[0]/base:.2f}, {rng[1]/base:.2f}]"
              f" (baseline {base:.5g})")
    print("\nterminated early:", {k: bool(v) for k, v in df["terminated_early"].items()},
          "| all finite:", bool(df["all_finite"].all()))
    refs = [("pid", "PID baseline")]
    if prefix != "td3_adaptive_pid_smoke":
        refs.append(("td3_adaptive_pid_smoke", "Stage 6a adaptive PID smoke (wide, 25 deg/2.1 dB box)"))
    for pre, name in refs:
        f = config.RESULTS_DIR / f"{pre}_metrics.csv"
        if f.exists():
            b = pd.read_csv(f, index_col=0)
            print(f"\n{name} (same scenarios, {f.name})")
            print(b[[c for c in cols if c in b.columns]].T)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--name", default=config.ADAPT_MODEL_NAME)
    p.add_argument("--zero-policy", action="store_true", help="evaluate a=0 (baseline PID)")
    a = p.parse_args(argv)
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    model = None if a.zero_policy else load_model(a.name)
    prefix = "adaptive_pid" if a.name == config.ADAPT_MODEL_NAME else a.name
    if a.zero_policy:
        prefix = "adaptive_pid_zero"
    _, df = evaluate(model, save_prefix=prefix)
    print_comparison(df, f"Adaptive PID ({'zero action' if a.zero_policy else a.name})", prefix)
    return df


if __name__ == "__main__":
    main()
