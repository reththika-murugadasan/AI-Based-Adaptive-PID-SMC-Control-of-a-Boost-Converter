"""Plots for the final five-controller comparison (Agg backend, PNG output).

``store[controller][scenario]`` is a list of series dicts (one per seed for the RL
controllers, a single one for the fixed controllers).  RL curves show the mean over seeds
with a min-max band so the seed-to-seed spread is visible.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config

# Okabe-Ito (colour-blind safe); PID family blue, SMC family orange/red, direct TD3 green.
STYLE = {
    "fixed_pid":    dict(color="#0072B2", ls="-",  label="Fixed PID"),
    "fixed_smc":    dict(color="#D55E00", ls="-",  label="Fixed SMC"),
    "direct_td3":   dict(color="#009E73", ls="-",  label="Direct TD3"),
    "adaptive_pid": dict(color="#56B4E9", ls="--", label="TD3 adaptive PID (6b)"),
    "adaptive_smc": dict(color="#E69F00", ls="--", label="TD3 adaptive SMC"),
}
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 150})

SIGNAL = {
    "vout": ("Vout [V]", lambda s: s["vout"]),
    "error": ("Vref - Vout [V]", lambda s: s["vref"] - s["vout"]),
    "duty": ("duty cycle D [-]", lambda s: s["duty"]),
    "il": ("inductor current IL [A]", lambda s: s["il"]),
}


def _stack(series_list, fn):
    """(t, mean, lo, hi) over seeds. Runs that ended early (protection trip) are padded with NaN,
    so every seed contributes for as long as it actually ran."""
    n = max(len(s["t"]) for s in series_list)
    arr = np.full((len(series_list), n), np.nan)
    for i, s in enumerate(series_list):
        v = np.asarray(fn(s), float)
        arr[i, :len(v)] = v
    longest = max(series_list, key=lambda s: len(s["t"]))["t"]
    with np.errstate(all="ignore"), __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("ignore", RuntimeWarning)
        return longest, np.nanmean(arr, 0), np.nanmin(arr, 0), np.nanmax(arr, 0)


def comparison_figure(store, scenarios, key, fname, out_dir, t_dist=config.SCENARIO_T_DIST,
                      startup_ms=(0.0, 40.0), dist_ms=(-5.0, 45.0), ylims=None, show_ref=False):
    """Signal ``key`` for all controllers. Row 1: startup; row 2: window around the disturbance."""
    ylabel, fn = SIGNAL[key]
    names = list(scenarios)
    fig, axs = plt.subplots(2, len(names), figsize=(4.0 * len(names), 6.2), squeeze=False)
    for j, sc in enumerate(names):
        for ctrl, st in STYLE.items():
            runs = store.get(ctrl, {}).get(sc)
            if not runs:
                continue
            t, m, lo, hi = _stack(runs, fn)
            for i, win in enumerate((startup_ms, tuple(t_dist * 1e3 + np.array(dist_ms)))):
                ax = axs[i, j]
                ax.plot(t * 1e3, m, color=st["color"], ls=st["ls"], lw=1.3,
                        label=st["label"] if (i == 0 and j == 0) else None)
                if len(runs) > 1:
                    ax.fill_between(t * 1e3, lo, hi, color=st["color"], alpha=0.18, lw=0)
                for r in runs:                       # mark runs that ended early (protection trip)
                    if r.get("terminated"):
                        ax.plot(r["t"][-1] * 1e3, np.asarray(fn(r))[-1], "x", color=st["color"],
                                ms=6, mew=1.5)
                ax.set_xlim(*win)
        first = next(r for c in STYLE if store.get(c, {}).get(sc) for r in store[c][sc])
        if show_ref:
            for i in range(2):
                axs[i, j].plot(first["t"] * 1e3, first["vref"], "k:", lw=0.9)
        axs[0, j].set_title(f"{first['label']}\nstartup", fontsize=9)
        axs[1, j].set_title("around disturbance" if first.get("disturbed") else
                            "same window (no disturbance)", fontsize=9)
        axs[1, j].axvline(t_dist * 1e3, color="0.4", lw=0.8, ls="--")
        axs[1, j].set_xlabel("time [ms]")
        if ylims:
            for i in range(2):
                axs[i, j].set_ylim(*ylims[i])
    for i in range(2):
        axs[i, 0].set_ylabel(ylabel)
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False)
    fig.suptitle(f"{ylabel}: all controllers (RL: mean over seeds, band = min-max; x = run ended on a protection trip)")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / fname
    fig.savefig(path)
    plt.close(fig)
    return path


def parameter_figure(runs_by_scenario, names, base, bounds, fname, out_dir, title,
                     labels=None, t_dist=config.SCENARIO_T_DIST, colors=None):
    """Adaptive-controller parameter trajectories: one thin line per seed.

    ``runs_by_scenario[scenario]`` = list of series dicts holding t and each name in ``names``.
    """
    scen = list(runs_by_scenario)
    fig, axs = plt.subplots(len(names), len(scen), figsize=(3.9 * len(scen), 2.2 * len(names) + 1.2),
                            sharex=True, squeeze=False)
    seed_colors = colors or ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
    for j, sc in enumerate(scen):
        runs = runs_by_scenario[sc]
        for i, p in enumerate(names):
            ax = axs[i, j]
            for k, s in enumerate(runs):
                ax.plot(s["t"] * 1e3, s[p], lw=1.1, color=seed_colors[k % len(seed_colors)],
                        label=f"seed {s.get('seed', k)}" if (i == 0 and j == 0) else None)
            ax.axhline(base[p], color="k", ls="--", lw=0.8, label="baseline" if (i == 0 and j == 0) else None)
            for b in bounds[p]:
                ax.axhline(b, color="r", ls=":", lw=0.8)
            if runs[0].get("disturbed"):
                ax.axvline(t_dist * 1e3, color="0.4", ls="--", lw=0.8)
            if j == 0:
                ax.set_ylabel((labels or {}).get(p, p))
            if i == 0:
                ax.set_title(runs[0]["label"], fontsize=9)
            if i == len(names) - 1:
                ax.set_xlabel("time [ms]")
    axs[0, 0].legend(fontsize=7, ncol=2)
    fig.suptitle(title + "  (red dotted = allowed range)")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / fname
    fig.savefig(path)
    plt.close(fig)
    return path


def recovery_figure(df_runs, disturbance_scenarios, fname, out_dir):
    """Maximum deviation and recovery time after each disturbance, per controller.

    Bars = mean over seeds, whiskers = min-max over seeds.  A recovery bar drawn hatched at the
    top marks runs that never returned to +-2 % within the run (recovery undefined).
    """
    ctrls = [c for c in STYLE if c in set(df_runs["controller"])]
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.4))
    width = 0.8 / len(ctrls)
    for ax, (metric, ylabel) in zip(axs, (("max_dev_after_V", "max |Vout - Vref| after disturbance [V]"),
                                          ("recovery_time_ms", "time to stay within +-2 % [ms]"))):
        top = np.nanmax(df_runs[df_runs.scenario.isin(disturbance_scenarios)][metric].to_numpy(float))
        top = 1.0 if not np.isfinite(top) else top
        for k, ctrl in enumerate(ctrls):
            for j, sc in enumerate(disturbance_scenarios):
                sub = df_runs[(df_runs.controller == ctrl) & (df_runs.scenario == sc)][metric].to_numpy(float)
                x = j + (k - (len(ctrls) - 1) / 2) * width
                fin = sub[np.isfinite(sub)]
                if fin.size:
                    ax.bar(x, fin.mean(), width * 0.92, color=STYLE[ctrl]["color"], alpha=0.9,
                           label=STYLE[ctrl]["label"] if j == 0 else None)
                    if fin.size > 1:
                        ax.vlines(x, fin.min(), fin.max(), color="k", lw=1.0)
                if metric == "recovery_time_ms" and (sub.size - fin.size) > 0:
                    ax.bar(x, top * 1.12, width * 0.92, facecolor="none", edgecolor=STYLE[ctrl]["color"],
                           hatch="///", lw=0.8)
                    ax.text(x, top * 1.14, f"{sub.size - fin.size}/{sub.size}\nno rec.", ha="center",
                            va="bottom", fontsize=6)
        ax.set_xticks(range(len(disturbance_scenarios)))
        ax.set_xticklabels([s.replace("_", " ") for s in disturbance_scenarios])
        ax.set_ylabel(ylabel)
        if metric == "recovery_time_ms":
            ax.set_ylim(0, top * 1.35)
    axs[0].legend(fontsize=7, frameon=False)
    fig.suptitle("Disturbance response (mean over seeds for RL; whisker = min-max; hatched = never recovered)")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / fname
    fig.savefig(path)
    plt.close(fig)
    return path
