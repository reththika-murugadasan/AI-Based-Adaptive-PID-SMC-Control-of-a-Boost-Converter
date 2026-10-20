import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from controllers.smc import SlidingModeController, sat
from converter import BoostConverter
from experiments.run_pid import run_scenario


def test_sat_function():
    assert sat(-5) == -1 and sat(5) == 1 and sat(0.3) == 0.3 and sat(-0.3) == -0.3


def test_vout_is_non_minimum_phase_on_our_model():
    """Justifies the current surface: a duty INCREASE lowers dVout/dt at first."""
    c = BoostConverter()
    il, v = c.steady_state(0.5)
    c.reset(IL=il, Vout=v)
    x = c.get_state()
    dv_lo = c.derivatives(x, 0.50)[1]
    dv_hi = c.derivatives(x, 0.55)[1]
    assert dv_hi < dv_lo                      # immediate effect: wrong sign
    # ...while the current responds with the right sign
    assert c.derivatives(x, 0.55)[0] > c.derivatives(x, 0.50)[0]


def test_sliding_surface_value():
    smc = SlidingModeController(Kv=0.5, Ki=0.0)
    # e = 2 V -> Iref = 1 A ; IL = 3 A -> s = 2 A
    assert smc.sliding_surface(2.0, 3.0) == pytest.approx(2.0)
    assert smc.sliding_surface(2.0, 0.5) == pytest.approx(-0.5)


def test_current_reference_limits():
    smc = SlidingModeController(Kv=1.0, Ki=0.0, i_max=5.0)
    assert smc.sliding_surface(100.0, 0.0) == pytest.approx(-5.0)   # clipped to I_max
    assert smc.sliding_surface(-100.0, 0.0) == pytest.approx(0.0)   # never negative


def test_duty_limits_always_respected():
    smc = SlidingModeController()
    rng = np.random.default_rng(0)
    for _ in range(2000):
        d = smc.update(rng.uniform(0, 40), rng.uniform(0, 60), config.DT,
                       rng.uniform(-5, 30))
        assert config.DUTY_MIN <= d <= config.DUTY_MAX


def test_switching_direction():
    """s > 0 (current too high) lowers D; s < 0 raises it, around D_eq."""
    smc = SlidingModeController(Kv=0.0, Ki=0.0)
    d_eq = smc.equivalent_duty(24.0)
    assert smc.update(24.0, 24.0, config.DT, inductor_current=5.0) < d_eq
    smc.reset()
    smc.xi = 5.0                               # Iref = 5 A, IL = 0 -> s < 0
    assert smc.update(24.0, 24.0, config.DT, inductor_current=0.0) > d_eq


def test_boundary_layer_is_linear_inside_and_saturated_outside():
    smc = SlidingModeController(Kv=0.0, Ki=0.0, switching_gain=0.3, boundary_layer=0.8)
    d_eq = smc.equivalent_duty(24.0)
    assert smc.update(24.0, 24.0, config.DT, 0.4) == pytest.approx(d_eq - 0.3 * 0.5)
    assert smc.update(24.0, 24.0, config.DT, 5.0) == pytest.approx(d_eq - 0.3)


def test_boundary_layer_must_be_positive():
    with pytest.raises(ValueError):
        SlidingModeController(boundary_layer=0.0)


def test_integral_anti_windup():
    smc = SlidingModeController()
    for _ in range(40000):                      # 2 s of large error at saturation
        smc.update(24.0, 0.0, config.DT, 0.0)
    assert smc.xi <= 2 * smc.i_max + 1e-9
    n = 0                                       # recovers quickly after error flips
    while smc.update(24.0, 30.0, config.DT, 5.0) >= config.DUTY_MAX - 1e-9 and n < 2000:
        n += 1
    assert n < 2000


def test_non_finite_input_rejected():
    with pytest.raises(ValueError):
        SlidingModeController().update(24.0, float("nan"), config.DT, 0.0)


@pytest.mark.parametrize("kind,val", [(None, None), ("vin", 10.0), ("vin", 14.0),
                                      ("load", 10.0), ("load", 30.0)])
def test_closed_loop_scenarios(kind, val):
    r = run_scenario(kind, val, controller=SlidingModeController())
    assert np.all(np.isfinite(r["vout"]))
    assert r["vout"][-1] == pytest.approx(24.0, abs=0.05)         # regulates
    assert r["il"].min() >= -1e-9                                  # diode: no reverse IL
    assert r["duty"].min() >= config.DUTY_MIN and r["duty"].max() <= config.DUTY_MAX
    assert r["il"].max() < 1.5 * config.SMC_I_MAX                  # current limiting works


def test_chattering_is_small():
    r = run_scenario(None, None, controller=SlidingModeController())
    steady = r["duty"][-2000:]                  # last 100 ms
    assert np.max(np.abs(np.diff(steady))) < 1e-6   # no limit cycle at steady state


def test_robust_to_component_variation():
    """+-20 % L and C, Vref 20 and 28 V: must still regulate."""
    for L, C, vref in [(0.8e-3, 376e-6, 28.0), (1.2e-3, 564e-6, 20.0)]:
        conv_kw = dict(L=L, C=C, Vref=vref)
        r = run_scenario(None, None, controller=SlidingModeController(),
                         converter_kwargs=conv_kw)
        assert r["vout"][-1] == pytest.approx(vref, abs=0.1)
