"""Stage 2: run the conventional PID on the five required scenarios.

Each run: cold start (Vout = 0, IL = 0) -> regulate to Vref -> disturbance at
T_DIST -> run to T_END.  Metrics, time series and plots go to results/.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from analysis import metrics as M
from analysis import plots
from controllers.pid import PIDController
from converter import BoostConverter


def run_scenario(kind, value, controller=None, t_dist=config.SCENARIO_T_DIST,
                 t_end=config.SCENARIO_T_END, converter_kwargs=None):
    conv = BoostConverter(**(converter_kwargs or {}))
    conv.reset(IL=0.0, Vout=0.0)
    pid = controller or PIDController()
    pid.reset()
    n, kd = int(round(t_end / conv.timestep)), int(round(t_dist / conv.timestep))
    t, v, il, d, vr = (np.zeros(n) for _ in range(5))
    for k in range(n):
        if k == kd and kind == "vin": conv.set_input_voltage(value)
        if k == kd and kind == "load": conv.set_load(value)
        if getattr(pid, 'uses_current', False):
            duty = pid.update(conv.Vref, conv.Vout, conv.timestep, conv.IL)
        else:
            duty = pid.update(conv.Vref, conv.Vout)
        conv.step(duty)
        t[k], v[k], il[k], d[k], vr[k] = conv.time, conv.Vout, conv.IL, conv.duty, conv.Vref
    return dict(t=t, vout=v, il=il, duty=d, vref=vr, disturbed=kind is not None)


def compute_metrics(r, t_dist=config.SCENARIO_T_DIST):
    t, v, vref = r["t"], r["vout"], r["vref"][0]
    pre = t < t_dist
    tp, vp = t[pre], v[pre]
    out = dict(
        rise_time_ms=M.rise_time(tp, vp, vref, v0=0.0) * 1e3,
        settling_time_ms=M.settling_time(tp, vp, vref, tol=0.02, t0=0.0) * 1e3,
        overshoot_pct=M.overshoot_pct(tp, vp, vref, v0=0.0),
        undershoot_V=M.undershoot_after_rise(tp, vp, vref),
        ss_error_V=M.steady_state_error(tp, vp, vref, window=0.01),   # before disturbance
        rmse_V=M.rmse(t, v, vref),                                    # whole run
        max_dev_after_V=M.max_deviation(t, v, vref, t_start=t_dist),
    )
    if r["disturbed"]:
        out["recovery_time_ms"] = M.recovery_time(t, v, vref, t_dist) * 1e3
        out["ss_error_final_V"] = M.steady_state_error(t, v, vref, window=0.01)
    else:
        out["recovery_time_ms"] = float("nan")
        out["ss_error_final_V"] = M.steady_state_error(t, v, vref, window=0.01)
    out["duty_variation"] = M.duty_variation(r["duty"])
    out["max_IL_A"] = M.max_current(r["il"])
    out["min_IL_A"] = float(r["il"].min())
    out["D_min"], out["D_max"] = float(r["duty"].min()), float(r["duty"].max())
    return out


def main():
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    results, rows = {}, []
    for name, (kind, val, label) in config.PID_SCENARIOS.items():
        r = run_scenario(kind, val)
        r["label"] = label
        results[name] = r
        m = compute_metrics(r)
        rows.append(dict(scenario=name, **m))
        pd.DataFrame({k: r[k] for k in ("t", "vout", "il", "duty", "vref")}).to_csv(
            config.DATA_DIR / f"pid_{name}.csv", index=False)
    df = pd.DataFrame(rows).set_index("scenario")
    df.to_csv(config.RESULTS_DIR / "pid_metrics.csv")
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(f"PID gains: Kp={config.PID_KP:g} Ki={config.PID_KI:g} Kd={config.PID_KD:g}")
    print(df.round(3).T)

    td = config.SCENARIO_T_DIST
    plots.scenario_grid(results, "vout", "PID: Vref vs Vout", "pid_vref_vs_vout.png",
                        t_dist=td, show_ref=True)
    plots.scenario_grid(results, "error", "PID: voltage error", "pid_error.png", t_dist=td,
                        ylim=(-6, 6))
    plots.scenario_grid(results, "duty", "PID: duty cycle", "pid_duty.png", t_dist=td)
    plots.scenario_grid(results, "il", "PID: inductor current", "pid_inductor_current.png",
                        t_dist=td)
    print("\nSaved figures to", config.FIGURES_DIR)
    return df


if __name__ == "__main__":
    main()
