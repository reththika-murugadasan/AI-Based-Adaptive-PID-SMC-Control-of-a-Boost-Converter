import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from converter import BoostConverter


def run(conv, duty, t_end):
    for _ in range(int(round(t_end / conv.timestep))):
        conv.step(duty)
    return conv


def test_reset_state():
    c = BoostConverter()
    c.step(0.5)
    s = c.reset()
    assert s.shape == (2,)
    assert s[0] == 0.0 and s[1] == config.VIN and c.time == 0.0


def test_ideal_duty_is_half():
    assert BoostConverter.ideal_duty(12, 24) == pytest.approx(0.5)


@pytest.mark.parametrize("duty", [0.3, 0.5, 0.6])
def test_steady_state_matches_theory(duty):
    c = run(BoostConverter(), duty, 1.0)   # 1 s >> settling time
    th = c.steady_state(duty)
    assert c.Vout == pytest.approx(th[1], rel=1e-3)
    assert c.IL == pytest.approx(th[0], rel=1e-3)


def test_equilibrium_has_zero_derivative():
    c = BoostConverter()
    x = c.steady_state(0.5)
    assert np.allclose(c.derivatives(x, 0.5), 0.0, atol=1e-6)


def test_duty_clamped():
    c = BoostConverter()
    c.step(-3.0); assert c.duty == config.DUTY_MIN
    c.step(7.0);  assert c.duty == config.DUTY_MAX


def test_nan_duty_rejected():
    with pytest.raises(ValueError):
        BoostConverter().step(float("nan"))


def test_states_finite_and_bounded_under_extreme_inputs():
    c = BoostConverter()
    rng = np.random.default_rng(0)
    for _ in range(20000):
        c.step(rng.uniform(-1, 2))
    assert np.all(np.isfinite(c.get_state())) and c.Vout < 200


def test_setters():
    c = BoostConverter()
    c.set_input_voltage(10); c.set_load(30); c.set_reference(20)
    assert (c.Vin, c.R, c.Vref) == (10, 30, 20)
    with pytest.raises(ValueError):
        c.set_load(0)


def test_load_increase_lowers_output_open_loop_current_rises():
    c = run(BoostConverter(), 0.5, 0.5)
    i0 = c.IL
    c.set_load(10.0)
    run(c, 0.5, 0.5)
    assert c.IL > i0


def test_inductor_current_never_negative_open_loop():
    for d in (0.3, 0.5, 0.6):
        c, lo = BoostConverter(), 0.0
        for _ in range(4000):
            c.step(d); lo = min(lo, c.IL)
        assert lo >= -1e-12


def test_inductor_current_never_negative_random_duty():
    c = BoostConverter()
    rng = np.random.default_rng(1)
    for _ in range(20000):
        c.step(rng.uniform(0, 1)); assert c.IL >= -1e-12


def test_ccm_only_reproduces_stage1_defect():
    c = BoostConverter(conduction_mode="ccm_only")
    lo = 0.0
    for _ in range(4000):
        c.step(0.5); lo = min(lo, c.IL)
    assert lo < -1.0


def test_dcm_steady_state_matches_closed_form():
    c = BoostConverter(R=500.0)
    run(c, 0.5, 4.0)
    th = c.steady_state(0.5)
    assert c.mode == "DCM"
    assert c.Vout == pytest.approx(th[1], rel=1e-3)
    assert c.IL == pytest.approx(th[0], rel=2e-3)


def test_dcm_ccm_boundary_continuity():
    """IL(DCM formula) equals the critical current when Vout = Vin/(1-D)."""
    c = BoostConverter()
    D = 0.4
    assert c.il_dcm(c.Vin / (1 - D), D) == pytest.approx(c.il_critical(D))


def test_substep_convergence():
    a, b = BoostConverter(substeps=5), BoostConverter(substeps=40)
    for _ in range(2000):
        a.step(0.5); b.step(0.5)
    assert a.Vout == pytest.approx(b.Vout, abs=0.05)
