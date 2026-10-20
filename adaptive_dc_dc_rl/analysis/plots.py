"""Plot helpers (Agg backend; every function saves a PNG and returns the path)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config

plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 100, "savefig.dpi": 150})

_YLAB = {"vout": "Vout [V]", "error": "Voltage error Vref - Vout [V]",
         "duty": "Duty cycle D [-]", "il": "Inductor current IL [A]"}


def scenario_grid(results, key, title, fname, t_dist=None, show_ref=False,
                  ylim=None, xlim=None):
    """One panel per scenario for signal ``key`` ('vout','error','duty','il').

    ``results``: {name: dict(t, vout, il, duty, vref, label)}.
    """
    names = list(results)
    n = len(names)
    cols = 3 if n > 4 else 2
    rows = -(-n // cols)
    fig, axs = plt.subplots(rows, cols, figsize=(5.2 * cols, 3.3 * rows), sharex=True)
    axs = axs.ravel()
    for ax, name in zip(axs, names):
        r = results[name]
        y = {"vout": r["vout"], "error": r["vref"] - r["vout"],
             "duty": r["duty"], "il": r["il"]}[key]
        ax.plot(r["t"] * 1e3, y, lw=1.4, label=_YLAB[key].split(" [")[0])
        if show_ref:
            ax.plot(r["t"] * 1e3, r["vref"], "k--", lw=1, label="Vref")
            ax.axhspan(0.98 * r["vref"][-1], 1.02 * r["vref"][-1], color="g", alpha=0.12,
                       label="+-2 %")
        if key == "duty":
            ax.axhline(config.DUTY_MAX, color="r", lw=0.6, ls=":")
            ax.axhline(config.DUTY_MIN, color="r", lw=0.6, ls=":")
        if t_dist is not None and r.get("disturbed", True):
            ax.axvline(t_dist * 1e3, color="orange", lw=1, ls="--")
        ax.set_title(r["label"], fontsize=10)
        ax.set_ylabel(_YLAB[key])
        if ylim: ax.set_ylim(*ylim)
        if xlim: ax.set_xlim(*xlim)
    for ax in axs[n:]:
        ax.axis("off")
    for ax in axs[:n]:
        if ax.get_subplotspec().is_last_row() or True:
            ax.set_xlabel("time [ms]")
    axs[0].legend(fontsize=8, loc="best")
    fig.suptitle(title)
    fig.tight_layout()
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / fname
    fig.savefig(path)
    plt.close(fig)
    return path


def gain_grid(results, fname, base, title="Adaptive PID gains", t_dist=None, bounds=None):
    """Kp / Ki / Kd trajectories (one row per gain, one column per scenario).

    ``results[name]`` needs t, Kp, Ki, Kd (arrays) and label. ``base``/``bounds``:
    dicts keyed by gain name (baseline value, (lo, hi)).
    """
    names = list(results)
    fig, axs = plt.subplots(3, len(names), figsize=(4.2 * len(names), 7.5), sharex=True,
                            squeeze=False)
    for j, name in enumerate(names):
        r = results[name]
        for i, g in enumerate(("Kp", "Ki", "Kd")):
            ax = axs[i, j]
            ax.plot(r["t"] * 1e3, r[g], lw=1.4)
            ax.axhline(base[g], color="k", ls="--", lw=0.8, label="baseline")
            if bounds:
                for b in bounds[g]:
                    ax.axhline(b, color="r", ls=":", lw=0.8)
            if t_dist is not None and r.get("disturbed", True):
                ax.axvline(t_dist * 1e3, color="orange", lw=1, ls="--")
            if j == 0:
                ax.set_ylabel(f"{g}")
            if i == 0:
                ax.set_title(r["label"], fontsize=9)
            if i == 2:
                ax.set_xlabel("time [ms]")
    axs[0, 0].legend(fontsize=8)
    fig.suptitle(title + "  (red dotted = safe range)")
    fig.tight_layout()
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / fname
    fig.savefig(path)
    plt.close(fig)
    return path


def param_grid(results, fname, names, base, bounds=None, title="Adaptive SMC parameters",
               t_dist=None, labels=None):
    """Parameter trajectories (one row per parameter, one column per scenario).

    ``results[name]`` needs t, a series for every parameter in ``names``, and label.
    """
    scen = list(results)
    fig, axs = plt.subplots(len(names), len(scen), figsize=(4.2 * len(scen), 2.5 * len(names) + 1),
                            sharex=True, squeeze=False)
    for j, sname in enumerate(scen):
        r = results[sname]
        for i, p in enumerate(names):
            ax = axs[i, j]
            ax.plot(r["t"] * 1e3, r[p], lw=1.4)
            ax.axhline(base[p], color="k", ls="--", lw=0.8, label="baseline")
            if bounds:
                for b in bounds[p]:
                    ax.axhline(b, color="r", ls=":", lw=0.8)
            if "recovering" in r and np.any(r["recovering"] > 0):
                ax.fill_between(r["t"] * 1e3, *ax.get_ylim(), where=r["recovering"] > 0,
                                color="orange", alpha=0.15)
            if t_dist is not None and r.get("disturbed", True):
                ax.axvline(t_dist * 1e3, color="orange", lw=1, ls="--")
            if j == 0:
                ax.set_ylabel((labels or {}).get(p, p))
            if i == 0:
                ax.set_title(r["label"], fontsize=9)
            if i == len(names) - 1:
                ax.set_xlabel("time [ms]")
    axs[0, 0].legend(fontsize=8)
    fig.suptitle(title + "  (red dotted = safe range, orange shade = baseline recovery)")
    fig.tight_layout()
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / fname
    fig.savefig(path)
    plt.close(fig)
    return path
