"""Final evaluation: ALL FIVE controllers on exactly the same five scenarios.

    python experiments/final_evaluation.py --tag final            # after training with --tag final
    python experiments/final_evaluation.py --tag pipeline         # pipeline check models

Controllers
    fixed_pid, fixed_smc          deterministic baselines (one run per scenario)
    direct_td3                    RL, one model per seed
    adaptive_pid, adaptive_smc    RL, one model per seed (bounded parameter adaptation)

Every run: cold start (Vout = 0, IL = 0), Vin = 12 V, R = 20 ohm, Vref = 24 V, disturbance at
150 ms, 300 ms total, series recorded at every 50 us converter step.  RL policies are evaluated
deterministically.  The five scenarios are NOT the training distribution (training uses
randomised conditions, see config.ENV_*).

Outputs (results/final[_<tag>]/):
    final_metrics_by_run.csv   one row per (controller, seed, scenario)  <- the combined CSV
    final_summary.csv          mean / std / min / max over seeds, long format
    final_report.md            the same numbers as readable tables (no rankings or scores)
    run_metadata.json          models used (sha256), seeds, settings
    figures/*.png
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from analysis import final_plots as fp
from controllers.pid import PIDController
from controllers.smc import SlidingModeController
from experiments.run_pid import compute_metrics, run_scenario as run_fixed_scenario

CONTROLLERS = {                 # key -> (label, RL key in config.FINAL_MODEL_BASES or None)
    "fixed_pid": ("Fixed PID", None),
    "fixed_smc": ("Fixed SMC", None),
    "direct_td3": ("Direct TD3", "direct"),
    "adaptive_pid": ("TD3 adaptive PID (6b)", "adaptive-pid"),
    "adaptive_smc": ("TD3 adaptive SMC", "adaptive-smc"),
}
METRICS = ["rise_time_ms", "settling_time_ms", "overshoot_pct", "undershoot_V", "ss_error_V",
           "ss_error_final_V", "rmse_V", "max_dev_after_V", "recovery_time_ms",
           "duty_variation", "max_IL_A", "min_IL_A", "protection_trip"]
METRIC_NOTES = {
    "rise_time_ms": "10-90 % rise time of the cold start [ms] (dominated by diode pre-charge, not control)",
    "settling_time_ms": "cold start: time after which Vout stays within +-2 % of Vref [ms]",
    "overshoot_pct": "cold-start overshoot above Vref [% of the step]",
    "undershoot_V": "deepest dip below Vref after first reaching 90 % [V]",
    "ss_error_V": "mean error over the 10 ms before the disturbance [V]",
    "ss_error_final_V": "mean error over the last 10 ms of the run [V]",
    "rmse_V": "RMS voltage error over the whole run [V]",
    "max_dev_after_V": "maximum |Vout - Vref| after the disturbance time [V]",
    "recovery_time_ms": "time after the disturbance until Vout stays within +-2 % [ms] (undefined = never recovered, or no disturbance)",
    "duty_variation": "mean |change of duty| per 50 us step",
    "max_IL_A": "maximum inductor current [A]",
    "min_IL_A": "minimum inductor current [A]",
    "protection_trip": "1 if the run hit a protection limit (Vout > %g V, IL > %g A, negative or non-finite state)" % (
        config.ENV_V_MAX, config.ENV_I_MAX),
}


# ---------------------------------------------------------------- helpers
def model_path(rl_key, tag, seed, models_dir):
    return Path(models_dir) / f"{config.FINAL_MODEL_BASES[rl_key]}_{tag}_seed{seed}.zip"


def protection_trip(series):
    """(tripped, reason) judged identically for every controller, from the recorded series."""
    v, il = np.asarray(series["vout"], float), np.asarray(series["il"], float)
    if not (np.all(np.isfinite(v)) and np.all(np.isfinite(il))):
        return True, "non-finite state"
    if v.max() > config.ENV_V_MAX:
        return True, "over-voltage"
    if il.max() > config.ENV_I_MAX:
        return True, "over-current"
    if v.min() < -1e-6 or il.min() < -1e-6:
        return True, "negative state"
    if series.get("terminated") and series.get("failure"):
        return True, str(series["failure"])
    return False, ""


def _rl_module(rl_key):
    if rl_key == "direct":
        from training import evaluate_td3 as mod
    elif rl_key == "adaptive-pid":
        from training import evaluate_td3_adaptive_pid as mod
    else:
        from training import evaluate_td3_adaptive_smc as mod
    return mod


def run_one(ctrl, kind, value, model, t_dist, t_end):
    """One scenario for one controller (model = loaded SB3 model for RL controllers)."""
    rl_key = CONTROLLERS[ctrl][1]
    if ctrl == "fixed_pid":
        return run_fixed_scenario(kind, value, controller=PIDController(), t_dist=t_dist, t_end=t_end)
    if ctrl == "fixed_smc":
        return run_fixed_scenario(kind, value, controller=SlidingModeController(), t_dist=t_dist,
                                  t_end=t_end)
    return _rl_module(rl_key).run_scenario(model, kind, value, t_dist=t_dist, t_end=t_end)


def run_metrics(series, t_dist):
    m = compute_metrics(series, t_dist=t_dist)
    il = np.asarray(series["il"], float)
    m["max_IL_A"], m["min_IL_A"] = float(il.max()), float(il.min())
    tripped, reason = protection_trip(series)
    m["protection_trip"] = int(tripped)
    m["trip_reason"] = reason
    m["supervisor_events"] = int(series.get("recovery_events", 0))
    m["episode_return"] = float(series["episode_return"]) if "episode_return" in series else (
        float(np.sum(series["reward"])) if "reward" in series else np.nan)
    m["n_samples"] = int(len(series["t"]))
    return m


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --------------------------------------------------------------- collect
def collect(tag, seeds, models_dir, t_dist, t_end, controllers=None):
    """Run everything. Returns (df_runs, store, meta)."""
    controllers = controllers or list(CONTROLLERS)
    missing = [str(model_path(CONTROLLERS[c][1], tag, s, models_dir))
               for c in controllers if CONTROLLERS[c][1] for s in seeds
               if not model_path(CONTROLLERS[c][1], tag, s, models_dir).exists()]
    if missing:
        raise FileNotFoundError("missing trained models:\n  " + "\n  ".join(missing) +
                                f"\ntrain them first: python experiments/train_final.py --tag {tag} "
                                f"--seeds {' '.join(map(str, seeds))}")
    rows, store, meta_models = [], {c: {} for c in controllers}, {}
    for ctrl in controllers:
        label, rl_key = CONTROLLERS[ctrl]
        for seed in (seeds if rl_key else [None]):
            model = None
            if rl_key:
                path = model_path(rl_key, tag, seed, models_dir)
                model = _rl_module(rl_key).load_model(path)
                meta_models[f"{ctrl}_seed{seed}"] = dict(file=path.name, sha256=sha256(path))
            for scen, (kind, val, scen_label) in config.PID_SCENARIOS.items():
                series = run_one(ctrl, kind, val, model, t_dist, t_end)
                series.update(label=scen_label, seed=seed)
                m = run_metrics(series, t_dist)
                rows.append(dict(controller=ctrl, label=label, seed="" if seed is None else seed,
                                 scenario=scen, **m))
                store[ctrl].setdefault(scen, []).append(series)
    df = pd.DataFrame(rows)
    return df, store, meta_models


# ------------------------------------------------------------ statistics
def summarise(df_runs, metrics=METRICS):
    """Long-format statistics over seeds (fixed controllers: n = 1)."""
    out = []
    for (ctrl, scen), g in df_runs.groupby(["controller", "scenario"], sort=False):
        for m in metrics:
            v = g[m].to_numpy(float)
            fin = v[np.isfinite(v)]
            out.append(dict(
                controller=ctrl, scenario=scen, metric=m, n=len(v), n_undefined=int(len(v) - len(fin)),
                mean=fin.mean() if fin.size else np.nan,
                std=(fin.std(ddof=1) if fin.size > 1 else 0.0) if fin.size else np.nan,
                min=fin.min() if fin.size else np.nan, max=fin.max() if fin.size else np.nan))
    return pd.DataFrame(out)


def _fmt(v, digits=4):
    if not np.isfinite(v):
        return "n/a"
    return f"{v:.{digits}g}"


def report_markdown(df_runs, summary, tag, seeds, t_dist, t_end):
    lines = [f"# Final evaluation report ({tag})", "",
             f"Seeds for RL controllers: {list(seeds)} (sample std, n = {len(seeds)}). "
             f"Scenarios: {', '.join(config.PID_SCENARIOS)}. Disturbance at {t_dist * 1e3:g} ms, "
             f"run length {t_end * 1e3:g} ms, 50 us sampling. RL policies are deterministic.", "",
             "Measured values only: no ranking, scoring or overall winner is computed. "
             "`n/a` = undefined (e.g. no disturbance in the nominal scenario, or Vout never "
             "re-entered the +-2 % band); `[k/n undefined]` = undefined for k of n seeds.", ""]
    scen = list(config.PID_SCENARIOS)
    for m in METRICS:
        lines += [f"## {m}", "", f"_{METRIC_NOTES[m]}_", "",
                  "| controller | " + " | ".join(scen) + " |", "|---|" + "---|" * len(scen)]
        for ctrl, (label, rl) in CONTROLLERS.items():
            cells = []
            for sc in scen:
                r = summary[(summary.controller == ctrl) & (summary.scenario == sc)
                            & (summary.metric == m)]
                if r.empty:
                    cells.append("-"); continue
                r = r.iloc[0]
                if r.n == 1:
                    cells.append(_fmt(r["mean"]))
                else:
                    c = f"{_fmt(r['mean'])} ± {_fmt(r['std'], 2)}" if np.isfinite(r["mean"]) else "n/a"
                    if r.n_undefined and np.isfinite(r["mean"]):
                        c += f" [{int(r.n_undefined)}/{int(r.n)} undefined]"
                    cells.append(c)
            if any(c != "-" for c in cells):
                lines.append(f"| {label} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["## protection trips and supervisor activity", "",
              "| controller | runs | protection trips | supervisor engagements (sum) |", "|---|---|---|---|"]
    for ctrl, (label, rl) in CONTROLLERS.items():
        g = df_runs[df_runs.controller == ctrl]
        if len(g):
            lines.append(f"| {label} | {len(g)} | {int(g.protection_trip.sum())} | "
                         f"{int(g.supervisor_events.sum())} |")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ plots
def make_plots(store, df_runs, out_dir, t_dist):
    scen = list(config.PID_SCENARIOS)
    figs = out_dir / "figures"
    paths = [
        fp.comparison_figure(store, scen, "vout", "final_vout_tracking.png", figs, t_dist,
                             show_ref=True, ylims=[(0, 36), (14, 30)]),
        fp.comparison_figure(store, scen, "error", "final_voltage_error.png", figs, t_dist,
                             ylims=[(-12, 26), (-6, 8)]),
        fp.comparison_figure(store, scen, "duty", "final_duty_cycle.png", figs, t_dist),
        fp.comparison_figure(store, scen, "il", "final_inductor_current.png", figs, t_dist),
        fp.recovery_figure(df_runs, [s for s in scen if s != "nominal"],
                           "final_disturbance_recovery.png", figs)]
    if "adaptive_pid" in store:
        base = dict(Kp=config.PID_KP, Ki=config.PID_KI, Kd=config.PID_KD)
        bounds = dict(Kp=config.ADAPT_KP_RANGE, Ki=config.ADAPT_KI_RANGE, Kd=config.ADAPT_KD_RANGE)
        paths.append(fp.parameter_figure(store["adaptive_pid"], ("Kp", "Ki", "Kd"), base, bounds,
                                         "final_adaptive_pid_gains.png", figs,
                                         "TD3 adaptive PID gains", t_dist=t_dist))
    if "adaptive_smc" in store:
        from controllers.adaptive_smc import default_ranges
        base = dict(Kv=config.SMC_KV, Ki=config.SMC_KI, K=config.SMC_GAIN, phi=config.SMC_BOUNDARY)
        paths.append(fp.parameter_figure(
            store["adaptive_smc"], ("Kv", "Ki", "K", "phi"), base, default_ranges(),
            "final_adaptive_smc_params.png", figs, "TD3 adaptive SMC parameters", t_dist=t_dist,
            labels=dict(Kv="Kv [A/V]", Ki="Ki [A/(V s)]", K="K", phi="phi [A]")))
    return paths


# ------------------------------------------------------------------- main
def run(tag=config.FINAL_TAG, seeds=config.FINAL_SEEDS, models_dir=None, out_dir=None,
        t_dist=config.SCENARIO_T_DIST, t_end=config.SCENARIO_T_END, plots=True, controllers=None):
    models_dir = Path(models_dir or config.MODELS_DIR)
    out_dir = Path(out_dir or config.RESULTS_DIR / ("final" if tag == config.FINAL_TAG else f"final_{tag}"))
    out_dir.mkdir(parents=True, exist_ok=True)
    df, store, meta_models = collect(tag, list(seeds), models_dir, t_dist, t_end, controllers)
    df.to_csv(out_dir / "final_metrics_by_run.csv", index=False)
    summary = summarise(df)
    summary.to_csv(out_dir / "final_summary.csv", index=False)
    (out_dir / "final_report.md").write_text(report_markdown(df, summary, tag, seeds, t_dist, t_end))
    import stable_baselines3, torch
    (out_dir / "run_metadata.json").write_text(json.dumps(dict(
        tag=tag, seeds=list(seeds), t_dist=t_dist, t_end=t_end, scenarios=list(config.PID_SCENARIOS),
        models=meta_models, baselines=dict(pid=[config.PID_KP, config.PID_KI, config.PID_KD],
                                           smc=[config.SMC_KV, config.SMC_KI, config.SMC_GAIN,
                                                config.SMC_BOUNDARY]),
        versions=dict(stable_baselines3=stable_baselines3.__version__, torch=torch.__version__,
                      numpy=np.__version__)), indent=2))
    figs = make_plots(store, df, out_dir, t_dist) if plots else []
    return df, summary, store, out_dir, figs


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--tag", default=config.FINAL_TAG)
    p.add_argument("--seeds", type=int, nargs="+", default=list(config.FINAL_SEEDS))
    p.add_argument("--models-dir", default=str(config.MODELS_DIR))
    p.add_argument("--out-dir", default=None)
    p.add_argument("--no-plots", action="store_true")
    a = p.parse_args(argv)
    df, summary, store, out_dir, figs = run(a.tag, a.seeds, a.models_dir, a.out_dir,
                                            plots=not a.no_plots)
    print((out_dir / "final_report.md").read_text())
    print(f"combined CSV : {out_dir / 'final_metrics_by_run.csv'}  ({len(df)} rows)")
    print(f"summary CSV  : {out_dir / 'final_summary.csv'}")
    for f in figs:
        print("figure       :", f)
    return df


if __name__ == "__main__":
    main()
