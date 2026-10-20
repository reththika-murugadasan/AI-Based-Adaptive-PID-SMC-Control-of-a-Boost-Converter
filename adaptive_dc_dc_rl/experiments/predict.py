"""Try the trained controllers on an operating point of your choice.

    python experiments/predict.py --vin 11 --load 15 --vref 26
    python experiments/predict.py --vin 12 --load 20 --vref 24 --step-vin 10      # Vin drops at 150 ms
    python experiments/predict.py --vin 12 --load 20 --vref 24 --step-load 10 --seed 1 --plot

All five controllers start cold (Vout = 0, IL = 0) and run for 300 ms. Fixed PID and SMC are
deterministic; the three RL controllers are loaded from models/*_<tag>_seed<seed>.zip and run
deterministically. The summary is measured numbers plus plain statements of what was observed;
it does not rank the controllers.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from controllers.pid import PIDController
from controllers.smc import SlidingModeController
from experiments import final_evaluation as fe
from experiments.run_pid import run_scenario as run_fixed

TRAIN_RANGE = dict(vin=(10.0, 14.0), load=(10.0, 30.0), vref=(20.0, 28.0))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vin", type=float, default=config.VIN, help="input voltage [V]")
    ap.add_argument("--load", type=float, default=config.R, help="load resistance [ohm]")
    ap.add_argument("--vref", type=float, default=config.VREF, help="target output voltage [V]")
    ap.add_argument("--step-vin", type=float, default=None, help="new Vin applied at --t-dist")
    ap.add_argument("--step-load", type=float, default=None, help="new load [ohm] applied at --t-dist")
    ap.add_argument("--t-dist", type=float, default=config.SCENARIO_T_DIST, help="disturbance time [s]")
    ap.add_argument("--t-end", type=float, default=config.SCENARIO_T_END, help="run length [s]")
    ap.add_argument("--tag", default="final", help="model tag (default: final)")
    ap.add_argument("--seed", type=int, default=0, help="which trained RL seed to load (0, 1 or 2)")
    ap.add_argument("--plot", action="store_true", help="save results/predict/predict.png")
    a = ap.parse_args()
    if a.step_vin is not None and a.step_load is not None:
        ap.error("use only one of --step-vin / --step-load")
    kind, value = (("vin", a.step_vin) if a.step_vin is not None else
                   ("load", a.step_load) if a.step_load is not None else (None, None))

    for name, val in (("vin", a.vin), ("load", a.load), ("vref", a.vref)):
        lo, hi = TRAIN_RANGE[name]
        if not lo <= val <= hi:
            print(f"WARNING: {name}={val:g} is outside the training range [{lo:g}, {hi:g}]; "
                  f"RL results there are extrapolation.")
    if a.vref <= a.vin:
        sys.exit("a boost converter needs Vref > Vin")

    # RL evaluators read these at call time; the fixed controllers take converter kwargs.
    config.VIN, config.R, config.VREF = a.vin, a.load, a.vref
    ckw = dict(Vin=a.vin, R=a.load, Vref=a.vref)

    rows, series_all = [], {}
    for ctrl, (label, rl_key) in fe.CONTROLLERS.items():
        if ctrl == "fixed_pid":
            s = run_fixed(kind, value, controller=PIDController(), t_dist=a.t_dist, t_end=a.t_end,
                          converter_kwargs=ckw)
        elif ctrl == "fixed_smc":
            s = run_fixed(kind, value, controller=SlidingModeController(), t_dist=a.t_dist,
                          t_end=a.t_end, converter_kwargs=ckw)
        else:
            mod = fe._rl_module(rl_key)
            path = fe.model_path(rl_key, a.tag, a.seed, config.MODELS_DIR)
            s = mod.run_scenario(mod.load_model(path), kind, value, t_dist=a.t_dist, t_end=a.t_end)
        m = fe.run_metrics(s, a.t_dist)
        m["final_v"] = float(np.mean(s["vout"][-int(0.01 / config.DT):]))
        series_all[ctrl] = s
        rows.append((label, m))

    print(f"\nOperating point: Vin={a.vin:g} V, R={a.load:g} ohm, Vref={a.vref:g} V "
          f"(ideal duty about {1 - a.vin / a.vref:.2f}); RL seed {a.seed}, tag '{a.tag}'")
    if kind:
        print(f"Disturbance at {a.t_dist * 1e3:g} ms: {kind} -> {value:g}")
    hdr = f"{'controller':24s}{'settle ms':>10s}{'overshoot%':>11s}{'Vout end V':>11s}{'max dev V':>10s}" \
          f"{'peak IL A':>10s}{'duty chg':>10s}  trip"
    print("\n" + hdr + "\n" + "-" * len(hdr))
    f = lambda v, d=2: "n/a" if not np.isfinite(v) else f"{v:.{d}f}"
    for label, m in rows:
        print(f"{label:24s}{f(m['settling_time_ms'], 1):>10s}{f(m['overshoot_pct']):>11s}"
              f"{f(m['final_v']):>11s}{f(m['max_dev_after_V']):>10s}{f(m['max_IL_A']):>10s}"
              f"{m['duty_variation'] * 1e4:>9.1f}e-4  {'YES: ' + m['trip_reason'] if m['protection_trip'] else 'no'}")

    print("\nWhat was observed:")
    for label, m in rows:
        err = m["final_v"] - a.vref
        if m["protection_trip"]:
            print(f"- {label}: hit a protection limit ({m['trip_reason']}).")
            continue
        s = (f"- {label}: ends {abs(err):.2f} V {'above' if err > 0 else 'below'} the target"
             if abs(err) > 0.02 * a.vref else f"- {label}: holds the target (final error {err:+.2f} V)")
        s += f"; settles in {m['settling_time_ms']:.1f} ms" if np.isfinite(m["settling_time_ms"]) \
            else "; never settles inside +-2 %"
        if m["duty_variation"] > 20e-4:
            s += "; duty cycle moves a lot (oscillation or heavy activity)"
        print(s + ".")
    print("\nNo ranking is implied: compare the columns that matter for your application.")

    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
        for (label, _), s in zip(rows, series_all.values()):
            ax[0].plot(np.asarray(s["t"]) * 1e3, s["vout"], label=label)
            ax[1].plot(np.asarray(s["t"]) * 1e3, s["il"])
            ax[2].plot(np.asarray(s["t"]) * 1e3, s["duty"])
        ax[0].axhline(a.vref, color="k", ls=":", lw=0.8)
        ax[0].set_ylabel("Vout [V]"); ax[1].set_ylabel("IL [A]"); ax[2].set_ylabel("duty")
        ax[2].set_xlabel("time [ms]"); ax[0].legend(fontsize=8)
        out = Path(config.RESULTS_DIR if hasattr(config, "RESULTS_DIR") else "results") / "predict"
        out.mkdir(parents=True, exist_ok=True)
        fig.tight_layout(); fig.savefig(out / "predict.png", dpi=130)
        print(f"plot saved to {out / 'predict.png'}")


if __name__ == "__main__":
    main()
