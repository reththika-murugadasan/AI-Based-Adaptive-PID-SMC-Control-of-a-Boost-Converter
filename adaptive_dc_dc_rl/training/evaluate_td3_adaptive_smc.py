"""Deterministic evaluation of the TD3 adaptive SMC on the five standard scenarios.

    python training/evaluate_td3_adaptive_smc.py --name td3_adaptive_smc_smoke

The agent acts every 200 us; the SMC inside runs at 50 us.  Series are recorded at EVERY
50 us converter step (info["trace"]) so the metrics have the same sampling as the PID, SMC
and adaptive-PID results.  ``--zero-policy`` evaluates a = 0, i.e. the baseline SMC itself.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from analysis import plots
from controllers.adaptive_smc import PARAMS
from environment.adaptive_smc_env import AdaptiveSMCEnv
from experiments.run_pid import compute_metrics

SERIES = ("t", "vout", "il", "duty", "il_max", "il_min", "Kv", "Ki", "K", "phi", "recovering")


def load_model(name=config.ASMC_MODEL_NAME):
    from stable_baselines3 import TD3
    path = Path(name) if str(name).endswith(".zip") else config.MODELS_DIR / f"{name}.zip"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - train first (training/train_td3_adaptive_smc.py)")
    return TD3.load(path, device="cpu")


def run_scenario(model, kind, value, t_dist=config.SCENARIO_T_DIST, t_end=config.SCENARIO_T_END,
                 seed=0):
    env = AdaptiveSMCEnv(max_steps=int(round(t_end / (config.DT * config.ENV_CONTROL_STEPS))),
                         disturbance_prob=0.0)
    env.record_trace = True
    step_d = int(round(t_dist / env.dt))
    dist = None if kind is None else dict(step=step_d, vin=value if kind == "vin" else None,
                                          load=value if kind == "load" else None)
    obs, info = env.reset(seed=seed, options=dict(vin=config.VIN, load=config.R, vref=config.VREF,
                                                  init="cold", disturbance=dist))
    cols = {k: [] for k in SERIES}
    reward, finite, failure, terminated, n_events, reasons = 0.0, True, None, False, 0, set()
    while True:
        action = np.zeros(4, np.float32) if model is None else model.predict(obs, deterministic=True)[0]
        obs, r, terminated, truncated, info = env.step(action)
        reward += r
        finite &= bool(np.all(np.isfinite(obs)) and np.isfinite(r))
        tr = info["trace"]
        for k in ("t", "vout", "il", "duty"):
            cols[k].extend(tr[k])
        cols["il_max"].append(info["IL_max"]); cols["il_min"].append(info["IL_min"])
        n = len(tr["t"])
        for p in PARAMS:
            cols[p].extend([info[p]] * n)
        cols["recovering"].extend([float(info["recovering"])] * n)
        n_events = info["recovery_events"]
        if info["recovery_reason"]:
            reasons.add(info["recovery_reason"])
        if terminated or truncated:
            failure = info["failure"]
            break
    out = {k: np.asarray(v) for k, v in cols.items()}
    out["vref"] = np.full_like(out["vout"], env.converter.Vref)
    out.update(disturbed=kind is not None, finite=finite, terminated=terminated,
               failure=failure, episode_return=reward, recovery_events=n_events,
               recovery_reasons=",".join(sorted(reasons)))
    return out


def evaluate(model, t_dist=config.SCENARIO_T_DIST, t_end=config.SCENARIO_T_END, save_prefix=None):
    results, rows = {}, []
    for name, (kind, val, label) in config.PID_SCENARIOS.items():
        r = run_scenario(model, kind, val, t_dist=t_dist, t_end=t_end)
        r["label"] = label
        results[name] = r
        m = compute_metrics(r, t_dist=t_dist)
        m["max_IL_A"], m["min_IL_A"] = float(r["il_max"].max()), float(r["il_min"].min())
        for p in PARAMS:
            m[f"{p}_min"], m[f"{p}_max"] = float(r[p].min()), float(r[p].max())
        m["recovery_events"] = int(r["recovery_events"])
        m["recovery_reasons"] = r["recovery_reasons"]
        m["recovering_fraction"] = float(r["recovering"].mean())
        m["episode_return"] = float(r["episode_return"])
        m["terminated_early"] = bool(r["terminated"])
        m["failure"] = r["failure"] or ""
        m["all_finite"] = bool(r["finite"])
        rows.append(dict(scenario=name, **m))
        if save_prefix:
            pd.DataFrame({k: r[k] for k in ("t", "vout", "il", "duty", "vref") + PARAMS
                          + ("recovering",)}).to_csv(config.DATA_DIR / f"{save_prefix}_{name}.csv",
                                                      index=False)
    df = pd.DataFrame(rows).set_index("scenario")
    if save_prefix:
        config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(config.RESULTS_DIR / f"{save_prefix}_metrics.csv")
        lab = "Adaptive SMC"
        plots.scenario_grid(results, "vout", f"{lab}: Vref vs Vout", f"{save_prefix}_vref_vs_vout.png",
                            t_dist=t_dist, show_ref=True)
        plots.scenario_grid(results, "error", f"{lab}: voltage error", f"{save_prefix}_error.png",
                            t_dist=t_dist, ylim=(-6, 6))
        plots.scenario_grid(results, "duty", f"{lab}: duty cycle", f"{save_prefix}_duty.png",
                            t_dist=t_dist)
        plots.scenario_grid(results, "il", f"{lab}: inductor current",
                            f"{save_prefix}_inductor_current.png", t_dist=t_dist)
        base = dict(Kv=config.SMC_KV, Ki=config.SMC_KI, K=config.SMC_GAIN, phi=config.SMC_BOUNDARY)
        from controllers.adaptive_smc import default_ranges
        plots.param_grid(results, f"{save_prefix}_params.png", PARAMS, base,
                         bounds=default_ranges(), t_dist=t_dist,
                         labels=dict(Kv="Kv [A/V]", Ki="Ki [A/(V s)]", K="K (switching gain)",
                                     phi="phi [A]"))
    return results, df


COLS = ["settling_time_ms", "overshoot_pct", "undershoot_V", "ss_error_final_V", "rmse_V",
        "max_dev_after_V", "recovery_time_ms", "duty_variation", "max_IL_A", "min_IL_A"]


def print_comparison(df, label, prefix=None):
    pd.set_option("display.width", 250, "display.max_columns", 30,
                  "display.float_format", "{:.4g}".format)
    print(f"\n{label} (rows = metrics, columns = scenarios)")
    print(df[COLS].T)
    print("\nsupervisor activity:")
    print(df[["recovery_events", "recovering_fraction", "recovery_reasons"]].T)
    print("\nparameters used during the runs (min .. max) vs allowed range:")
    from controllers.adaptive_smc import default_ranges
    base = dict(Kv=config.SMC_KV, Ki=config.SMC_KI, K=config.SMC_GAIN, phi=config.SMC_BOUNDARY)
    for p, (lo, hi) in default_ranges().items():
        print(f"  {p:>3}: used {df[p + '_min'].min():.5g} .. {df[p + '_max'].max():.5g} | "
              f"allowed [{lo:.5g}, {hi:.5g}] = baseline x [{lo/base[p]:.2f}, {hi/base[p]:.2f}] "
              f"(baseline {base[p]:.5g})")
    print("\nterminated early:", {k: bool(v) for k, v in df["terminated_early"].items()},
          "| all finite:", bool(df["all_finite"].all()))


def print_all_comparisons(df, adaptive_label="adaptive SMC (smoke)"):
    """Adaptive SMC against fixed SMC, fixed PID, direct-TD3 smoke and adaptive-PID 6b.

    Also writes results/stage7_comparison.csv (one row per controller x scenario).
    """
    refs = [("smc", "fixed SMC"), ("pid", "fixed PID"),
            ("td3_smoke_r2", "direct TD3 smoke (r2 reward)"),
            ("td3_adaptive_pid_smoke_6b", "adaptive PID 6b smoke")]
    frames = {adaptive_label: df[COLS]}
    for pre, name in refs:
        f = config.RESULTS_DIR / f"{pre}_metrics.csv"
        if f.exists():
            b = pd.read_csv(f, index_col=0)
            frames[name] = b[[c for c in COLS if c in b.columns]]
            print(f"\n{name}  [{f.name}]")
            print(frames[name].T)
        else:
            print(f"\n{name}: {f.name} not found")
    comp = pd.concat(frames, names=["controller", "scenario"])
    comp.to_csv(config.RESULTS_DIR / "stage7_comparison.csv")
    return comp


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--name", default=config.ASMC_MODEL_NAME)
    p.add_argument("--zero-policy", action="store_true", help="evaluate a=0 (baseline SMC)")
    p.add_argument("--no-compare", action="store_true")
    a = p.parse_args(argv)
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    model = None if a.zero_policy else load_model(a.name)
    prefix = "adaptive_smc_zero" if a.zero_policy else (
        "adaptive_smc" if a.name == config.ASMC_MODEL_NAME else a.name)
    _, df = evaluate(model, save_prefix=prefix)
    print_comparison(df, f"Adaptive SMC ({'zero action' if a.zero_policy else a.name})", prefix)
    if not a.no_compare:
        print_all_comparisons(df, f"adaptive SMC ({a.name})")
    return df


if __name__ == "__main__":
    main()
