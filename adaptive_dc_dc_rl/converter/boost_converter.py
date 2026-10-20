"""Cycle-averaged boost converter model with CCM *and* DCM (ideal diode).

States reported: x = [IL, Vout]  (IL = cycle-averaged inductor current)

CCM (diode always conducting during the off-time):
    dIL/dt   = (Vin - (1-D) Vout) / L
    dVout/dt = ((1-D) IL - Vout/R) / C
For constant D over a step this is linear affine, x' = A x + B, so it is
integrated EXACTLY with a 3x3 matrix exponential (scipy.linalg.expm).

DCM (inductor current reaches zero before the period ends; the diode blocks
reverse current).  Per switching period Ts, with ipk = Vin D Ts / L:
    d2   = D Vin / (Vout - Vin)                  (diode conduction fraction)
    IL   = ipk (D + d2) / 2 = Vin D^2 Ts Vout / (2 L (Vout - Vin))   (algebraic!)
    Id   = ipk d2 / 2       = Vin^2 D^2 Ts / (2 L (Vout - Vin))
    C dVout/dt = Id - Vout/R
The inductor is no longer a dynamic state: IL is a function of (D, Vout).
Only Vout is integrated (RK4, smooth 1-D ODE).

Mode logic (valley of the inductor ripple touches zero):
    DCM  iff  Vout > Vin/(1-D)  and  IL < Vin D Ts/(2L)
The first condition is d2 < 1-D, the second is the average-current form of
the CCM/DCM boundary.  Otherwise CCM.

Not modelled: ESR, diode drop, inductor resistance, switching ripple.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm

import config


class BoostConverter:
    def __init__(self, Vin=config.VIN, L=config.L, C=config.C, R=config.R,
                 Vref=config.VREF, timestep=config.DT, Ts=config.T_SW,
                 duty_min=config.DUTY_MIN, duty_max=config.DUTY_MAX,
                 conduction_mode=config.CONDUCTION_MODE,
                 substeps=config.SUBSTEPS):
        if conduction_mode not in ("auto", "ccm_only"):
            raise ValueError(f"unknown conduction_mode {conduction_mode!r}")
        self.Vin, self.L, self.C, self.R = float(Vin), float(L), float(C), float(R)
        self.Vref = float(Vref)
        self.timestep, self.Ts = float(timestep), float(Ts)
        self.duty_min, self.duty_max = float(duty_min), float(duty_max)
        self.conduction_mode = conduction_mode
        self.substeps = int(substeps)
        self.reset()

    # ------------------------------------------------------------ state
    def reset(self, IL=config.IL0, Vout=None):
        """Default: IL = 0, Vout = Vin (capacitor pre-charged via the diode)."""
        self.IL = float(IL)
        self.Vout = float(self.Vin if Vout is None else Vout)
        self.duty = 0.0
        self.time = 0.0
        self.mode = "CCM"
        return self.get_state()

    def get_state(self):
        return np.array([self.IL, self.Vout], dtype=np.float64)

    # --------------------------------------------------- parameter setters
    def set_input_voltage(self, Vin):
        if Vin <= 0:
            raise ValueError("Vin must be positive")
        self.Vin = float(Vin)

    def set_load(self, R):
        if R <= 0:
            raise ValueError("R must be positive")
        self.R = float(R)

    def set_reference(self, Vref):
        if Vref <= 0:
            raise ValueError("Vref must be positive")
        self.Vref = float(Vref)

    # ------------------------------------------------------- helper maths
    def clamp_duty(self, duty):
        return float(np.clip(duty, self.duty_min, self.duty_max))

    def ccm_derivatives(self, x, duty):
        IL, V = x
        d1 = 1.0 - duty
        return np.array([(self.Vin - d1 * V) / self.L,
                         (d1 * IL - V / self.R) / self.C])

    derivatives = ccm_derivatives   # Stage-1 name kept for compatibility

    def il_critical(self, duty):
        """Average IL at the CCM/DCM boundary (half the ripple): Vin D Ts / 2L."""
        return self.Vin * duty * self.Ts / (2.0 * self.L)

    def il_dcm(self, V, duty):
        return self.Vin * duty ** 2 * self.Ts * V / (2.0 * self.L * (V - self.Vin))

    def _dv_dcm(self, V, duty):
        Id = self.Vin ** 2 * duty ** 2 * self.Ts / (2.0 * self.L * (V - self.Vin))
        return (Id - V / self.R) / self.C

    def steady_state(self, duty):
        """Equilibrium (IL, Vout), CCM or DCM depending on K = 2L/(R Ts)."""
        K = 2.0 * self.L / (self.R * self.Ts)
        if K >= duty * (1.0 - duty) ** 2:                # CCM (K_crit = D(1-D)^2)
            return np.array([self.Vin / (self.R * (1 - duty) ** 2),
                             self.Vin / (1 - duty)])
        V = self.Vin * (1.0 + np.sqrt(1.0 + 4.0 * duty ** 2 / K)) / 2.0
        return np.array([self.il_dcm(V, duty), V])

    @staticmethod
    def ideal_duty(Vin, Vout):
        return 1.0 - Vin / Vout

    # ----------------------------------------------------------- integrators
    def _ccm_step(self, IL, V, duty, h):
        d1 = 1.0 - duty
        M = np.zeros((3, 3))                    # augmented [A B; 0 0]
        M[0, 1] = -d1 / self.L
        M[0, 2] = self.Vin / self.L
        M[1, 0] = d1 / self.C
        M[1, 1] = -1.0 / (self.R * self.C)
        xa = expm(M * h) @ np.array([IL, V, 1.0])
        return xa[0], xa[1]

    def _dcm_step(self, V, duty, h):
        f = lambda v: self._dv_dcm(v, duty)
        k1 = f(V); k2 = f(V + 0.5 * h * k1)
        k3 = f(V + 0.5 * h * k2); k4 = f(V + h * k3)
        return V + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

    def _substep(self, duty, h):
        in_dcm_region = (self.conduction_mode == "auto"
                         and self.Vout > self.Vin / (1.0 - duty)
                         and self.IL < self.il_critical(duty))
        if in_dcm_region:
            V = self._dcm_step(self.Vout, duty, h)
            if V > self.Vin / (1.0 - duty):
                self.Vout, self.IL, self.mode = V, self.il_dcm(V, duty), "DCM"
                return
            # left DCM inside this sub-step: fall through to CCM from the
            # DCM current at the start of the sub-step
            self.IL = self.il_dcm(self.Vout, duty)
        IL, V = self._ccm_step(self.IL, self.Vout, duty, h)
        self.mode = "CCM"
        if (self.conduction_mode == "auto" and V > self.Vin / (1.0 - duty)
                and IL < self.il_critical(duty)):
            IL, self.mode = self.il_dcm(V, duty), "DCM"   # diode blocked
        self.IL, self.Vout = IL, V

    def step(self, duty_cycle):
        """Advance one timestep with the given duty cycle; return [IL, Vout]."""
        if not np.isfinite(duty_cycle):
            raise ValueError("duty cycle is NaN/inf")
        duty = self.clamp_duty(duty_cycle)
        h = self.timestep / self.substeps
        for _ in range(self.substeps):
            self._substep(duty, h)
        if not (np.isfinite(self.IL) and np.isfinite(self.Vout)):
            raise FloatingPointError("converter state became non-finite")
        self.duty = duty
        self.time += self.timestep
        return self.get_state()
