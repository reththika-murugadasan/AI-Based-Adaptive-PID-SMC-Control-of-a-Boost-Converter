"""Stage 1b: validate the boost converter model.

 1. Open-loop D = 0.3 / 0.5 / 0.6 : steady state vs theory.
 2. Transient vs a switch-level (ideal switch + ideal diode) reference.
 3. Old CCM-only model vs new CCM/DCM model (shows the defect).
 4. A genuine DCM operating point (R = 500 ohm) vs the closed-form DCM gain.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config
from converter import BoostConverter
from validation.switched_reference import simulate_switched


def simulate(duty, t_end=config.OPEN_LOOP_T_END, **kw):
    conv = BoostConverter(**kw)
    n = int(round(t_end / conv.timestep))
    out = np.zeros((n + 1, 4))                       # t, IL, V, dcm flag
    out[0, 1:3] = conv.get_state()
    for k in range(n):
        conv.step(duty)
        out[k + 1] = (conv.time, conv.IL, conv.Vout, conv.mode == "DCM")
    return out


def main():
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    conv = BoostConverter()
    print(f"Ideal duty {config.VIN:g}->{config.VREF:g} V: D = "
          f"{BoostConverter.ideal_duty(config.VIN, config.VREF):.4f}")
    print(f"dt = {config.DT*1e6:.0f} us, substeps = {config.SUBSTEPS}, "
          f"K = 2L/(R Ts) = {2*config.L/(config.R*config.T_SW):.2f} "
          f"(steady state is CCM for all D since K > max K_crit = 4/27 = 0.148)\n")
    ok = True
    fig, ax = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
    hdr = (f"{'D':>4} | {'Vss sim':>8} {'Vss th':>8} | {'ILss sim':>8} {'ILss th':>8} |"
           f" {'minIL new':>9} {'minIL old':>9} | {'peakV new':>9} {'peakV ref':>9} |"
           f" {'RMS dV':>7} {'RMS dIL':>8} {'max dV':>7}")
    print(hdr); print("-" * len(hdr))
    for j, d in enumerate(config.OPEN_LOOP_DUTIES):
        new = simulate(d)
        old = simulate(d, conduction_mode="ccm_only")
        tr, ir, vr = simulate_switched(d, config.OPEN_LOOP_T_END, config.VIN,
                                       config.L, config.C, config.R, config.T_SW)
        # model value at the middle of each switching period vs reference average
        t = new[:, 0]
        v_m = np.interp(tr, t, new[:, 2]); i_m = np.interp(tr, t, new[:, 1])
        dv, di = v_m - vr, i_m - ir
        th = conv.steady_state(d)
        m = t > t[-1] - 5e-3
        v_ss, i_ss = new[m, 2].mean(), new[m, 1].mean()
        print(f"{d:4.1f} | {v_ss:8.3f} {th[1]:8.3f} | {i_ss:8.3f} {th[0]:8.3f} |"
              f" {new[:,1].min():9.3f} {old[:,1].min():9.3f} | {new[:,2].max():9.2f}"
              f" {vr.max():9.2f} | {np.sqrt((dv**2).mean()):7.3f}"
              f" {np.sqrt((di**2).mean()):8.3f} {abs(dv).max():7.3f}")
        ok &= abs(v_ss - th[1]) / th[1] < 1e-3 and abs(i_ss - th[0]) / th[0] < 1e-3
        ok &= new[:, 1].min() >= -1e-9                  # diode: no reverse current
        ok &= np.sqrt((dv ** 2).mean()) < 0.5 and np.sqrt((di ** 2).mean()) < 0.15
        ok &= abs(new[:, 2].max() - vr.max()) < 1.5
        ax[0, j].plot(t * 1e3, old[:, 2], color="0.7", label="old CCM-only")
        ax[0, j].plot(tr * 1e3, vr, "k--", lw=1, label="switched reference (cycle avg)")
        ax[0, j].plot(t * 1e3, new[:, 2], label="new CCM/DCM model")
        ax[1, j].plot(t * 1e3, old[:, 1], color="0.7")
        ax[1, j].plot(tr * 1e3, ir, "k--", lw=1)
        ax[1, j].plot(t * 1e3, new[:, 1], label="new")
        ax[1, j].axhline(0, color="r", lw=0.6)
        dcm = new[:, 3] > 0
        ax[1, j].fill_between(t * 1e3, -5, 15, where=dcm, color="orange", alpha=0.2,
                              label="DCM active")
        ax[0, j].set_title(f"D = {d}"); ax[1, j].set_xlabel("time [ms]")
        ax[0, j].axhline(th[1], color="g", lw=0.6, ls=":")
    ax[0, 0].set_ylabel("Vout [V]"); ax[1, 0].set_ylabel("IL [A] (cycle average)")
    ax[0, 0].legend(fontsize=7); ax[1, 0].legend(fontsize=7)
    for a in ax.ravel(): a.grid(alpha=0.3)
    fig.suptitle("Boost converter, open loop, Vin = 12 V, R = 20 ohm")
    fig.tight_layout()
    f1 = config.FIGURES_DIR / "stage1b_open_loop_validation.png"
    fig.savefig(f1, dpi=150)

    # ---- genuine DCM operating point (needs light load: K < D(1-D)^2)
    Rd, Dd = 500.0, 0.5
    cv = BoostConverter(R=Rd)
    th = cv.steady_state(Dd)
    mod = simulate(Dd, 0.4, R=Rd)
    tr, ir, vr = simulate_switched(Dd, 0.4, config.VIN, config.L, config.C, Rd, config.T_SW)
    # transient (0.4 s, not settled: RC = 0.235 s) vs switched reference
    v_end, i_end = mod[-1, 2], mod[-1, 1]
    print(f"\nDCM point R = {Rd:g} ohm, D = {Dd}: K = {2*config.L/(Rd*config.T_SW):.2f} < "
          f"{Dd*(1-Dd)**2:.3f}")
    print(f"  t = 0.4 s (still settling) model Vout {v_end:.3f} V, IL {i_end:.4f} A | "
          f"switched ref Vout {vr[-1]:.3f} V, IL {ir[-1]:.4f} A")
    # steady state (4 s >> RC) vs closed form  M = (1 + sqrt(1 + 4 D^2/K)) / 2
    long = simulate(Dd, 4.0, R=Rd)
    print(f"  t = 4 s  model Vout {long[-1,2]:.3f} V, IL {long[-1,1]:.4f} A | closed form "
          f"Vout {th[1]:.3f} V, IL {th[0]:.4f} A | mode {'DCM' if long[-1,3] else 'CCM'}")
    ok &= abs(v_end - vr[-1]) < 0.05 and abs(long[-1, 2] - th[1]) < 0.01
    ok &= abs(long[-1, 1] - th[0]) < 1e-3 and long[-1, 3] == 1
    fig2, a2 = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    a2[0].plot(mod[:, 0] * 1e3, mod[:, 2], label="model"); a2[0].plot(tr * 1e3, vr, "k--", label="switched ref")
    a2[1].plot(mod[:, 0] * 1e3, mod[:, 1]); a2[1].plot(tr * 1e3, ir, "k--")
    a2[0].set(ylabel="Vout [V]", title=f"DCM operating point, R = {Rd:g} ohm, D = {Dd}")
    a2[1].set(ylabel="IL avg [A]", xlabel="time [ms]"); a2[0].legend()
    for a in a2: a.grid(alpha=0.3)
    fig2.tight_layout(); fig2.savefig(config.FIGURES_DIR / "stage1b_dcm_point.png", dpi=150)

    print(f"\nSaved {f1} and stage1b_dcm_point.png")
    print("RESULT:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
