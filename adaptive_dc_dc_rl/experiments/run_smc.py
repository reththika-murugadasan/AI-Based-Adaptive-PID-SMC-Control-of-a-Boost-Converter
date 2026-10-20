"""Stage 3: run the sliding-mode controller on the same five scenarios as the PID
and print a PID-vs-SMC comparison.  Results go to results/ (smc_* files)."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from analysis import plots
from controllers.pid import PIDController
from controllers.smc import SlidingModeController
from experiments.run_pid import compute_metrics, run_scenario


def run_all(make_controller, prefix, label, save=True):
    results, rows = {}, []
    for name, (kind, val, lab) in config.PID_SCENARIOS.items():
        r = run_scenario(kind, val, controller=make_controller())
        r["label"] = lab
        results[name] = r
        rows.append(dict(scenario=name, **compute_metrics(r)))
        if save:
            pd.DataFrame({k: r[k] for k in ("t", "vout", "il", "duty", "vref")}).to_csv(
                config.DATA_DIR / f"{prefix}_{name}.csv", index=False)
    df = pd.DataFrame(rows).set_index("scenario")
    if save:
        df.to_csv(config.RESULTS_DIR / f"{prefix}_metrics.csv")
    return results, df


def main():
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250, "display.max_columns", 30,
                  "display.float_format", "{:.4g}".format)
    smc_res, smc = run_all(SlidingModeController, "smc", "SMC")
    pid_res, pid = run_all(PIDController, "pid", "PID", save=False)
    print(f"SMC: Kv={config.SMC_KV:g} Ki={config.SMC_KI:g} K={config.SMC_GAIN:g} "
          f"phi={config.SMC_BOUNDARY:g} A  I_max={config.SMC_I_MAX:g} A")
    print(smc.T)

    td = config.SCENARIO_T_DIST
    plots.scenario_grid(smc_res, "vout", "SMC: Vref vs Vout", "smc_vref_vs_vout.png",
                        t_dist=td, show_ref=True)
    plots.scenario_grid(smc_res, "error", "SMC: voltage error", "smc_error.png", t_dist=td,
                        ylim=(-6, 6))
    plots.scenario_grid(smc_res, "duty", "SMC: duty cycle", "smc_duty.png", t_dist=td)
    plots.scenario_grid(smc_res, "il", "SMC: inductor current", "smc_inductor_current.png",
                        t_dist=td)

    # PID vs SMC overlay (zoomed on the disturbance and on the startup)
    fig, axs = plt.subplots(2, 5, figsize=(22, 7))
    for j, name in enumerate(config.PID_SCENARIOS):
        for res, lab, c in ((pid_res, "PID", "C0"), (smc_res, "SMC", "C3")):
            r = res[name]
            axs[0, j].plot(r["t"] * 1e3, r["vout"], c, label=lab, lw=1.3)
            axs[1, j].plot(r["t"] * 1e3, r["il"], c, label=lab, lw=1.3)
        axs[0, j].axhline(24, color="k", ls="--", lw=0.8)
        axs[0, j].set_xlim(0, 30) if j == 0 else axs[0, j].set_xlim(td * 1e3 - 5, td * 1e3 + 40)
        axs[1, j].set_xlim(*axs[0, j].get_xlim())
        axs[0, j].set_ylim(14, 30) if j else axs[0, j].set_ylim(0, 36)
        axs[0, j].set_title(config.PID_SCENARIOS[name][2], fontsize=9)
        axs[1, j].set_xlabel("time [ms]")
    axs[0, 0].set_ylabel("Vout [V]"); axs[1, 0].set_ylabel("IL [A]")
    axs[0, 0].legend()
    fig.suptitle("PID vs SMC (startup zoom, then 45 ms around the disturbance)")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "pid_vs_smc.png", dpi=150)
    plt.close(fig)

    cols = ["settling_time_ms", "overshoot_pct", "undershoot_V", "rmse_V",
            "max_dev_after_V", "recovery_time_ms", "ss_error_final_V", "duty_variation",
            "max_IL_A", "min_IL_A"]
    cmp = pd.concat({"PID": pid[cols], "SMC": smc[cols]}, names=["controller"])
    cmp.to_csv(config.RESULTS_DIR / "pid_vs_smc.csv")
    print("\nPID vs SMC")
    for c in cols:
        t = pd.DataFrame({"PID": pid[c], "SMC": smc[c]}).T
        print(f"\n{c}"); print(t)


if __name__ == "__main__":
    main()
