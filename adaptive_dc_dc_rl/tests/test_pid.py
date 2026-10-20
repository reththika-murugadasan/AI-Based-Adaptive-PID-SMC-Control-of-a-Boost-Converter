import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from analysis import metrics as M
from controllers.pid import PIDController
from converter import BoostConverter


def make(**kw):
    p = dict(Kp=0.01, Ki=10.0, Kd=0.0, dt=1e-4, bias=0.5)
    p.update(kw)
    return PIDController(**p)


def test_zero_error_returns_bias():
    pid = make()
    for _ in range(50):
        assert pid.update(24.0, 24.0) == pytest.approx(0.5)


def test_error_response_sign():
    """Vout below Vref -> duty above bias; Vout above Vref -> below bias."""
    assert make(Ki=0).update(24.0, 22.0) > 0.5
    assert make(Ki=0).update(24.0, 26.0) < 0.5


def test_proportional_value():
    assert make(Ki=0).update(24.0, 23.0) == pytest.approx(0.5 + 0.01 * 1.0)


def test_output_saturation():
    pid = make(Kp=100.0)
    assert pid.update(24.0, 0.0) == config.DUTY_MAX
    pid.reset()
    assert pid.update(0.0, 24.0) == config.DUTY_MIN


def test_integral_grows_with_persistent_error():
    pid = make(Kp=0.0)
    d = [pid.update(24.0, 23.0) for _ in range(100)]
    assert d[-1] > d[0] >= 0.5
    assert d[-1] == pytest.approx(0.5 + 10.0 * 1.0 * 99e-4 + 0.0, rel=0.02)


def test_anti_windup_bounded_and_fast_recovery():
    """Saturate for a long time; integral state must stay bounded and the
    output must leave saturation soon after the error changes sign."""
    pid = make(Kp=0.05, Ki=50.0)
    for _ in range(20000):                       # 2 s of huge positive error
        pid.update(24.0, 0.0)
    assert pid.i_term <= config.DUTY_MAX - 0.5 + 1e-12
    n = 0
    while pid.update(24.0, 30.0) >= config.DUTY_MAX:   # error flips sign
        n += 1
        assert n < 500                           # leaves saturation within 50 ms
    assert n < 500


def test_no_windup_without_antiwindup_would_be_worse():
    """Compare with aw_gain=0 + no clamp effect: back-calculation unwinds faster."""
    def unwind_steps(aw):
        pid = make(Kp=0.05, Ki=50.0, aw_gain=aw)
        pid.i_term = 0.0
        for _ in range(3000):
            pid.update(24.0, 0.0)
        n = 0
        while pid.update(24.0, 30.0) >= config.DUTY_MAX and n < 100000:
            n += 1
        return n
    assert unwind_steps(500.0) <= unwind_steps(0.0)


def test_derivative_filter_and_no_kick_on_reference_step():
    pid = make(Kp=0.0, Ki=0.0, Kd=1e-3, deriv_tau=1e-3)
    pid.update(24.0, 24.0)
    d = pid.update(30.0, 24.0)                   # reference step, measurement flat
    assert d == pytest.approx(0.5)               # derivative on measurement: no kick


def test_derivative_reacts_to_measurement_ramp():
    pid = make(Kp=0.0, Ki=0.0, Kd=1e-3, deriv_tau=1e-4)
    pid.update(24.0, 20.0)
    out = [pid.update(24.0, 20.0 + 0.1 * k) for k in range(1, 30)]
    assert out[-1] < 0.5                          # rising output -> negative D action


def test_gain_change_is_bumpless():
    pid = make()
    for _ in range(200):
        pid.update(24.0, 23.0)
    before = pid.i_term
    pid.set_gains(Ki=100.0)
    assert pid.i_term == before


def test_non_finite_input_rejected():
    with pytest.raises(ValueError):
        make().update(24.0, float("nan"))


def test_closed_loop_regulates_nominal_point():
    from experiments.run_pid import run_scenario
    r = run_scenario(None, None, controller=PIDController(), t_end=0.3)
    assert np.all(np.isfinite(r["vout"]))
    assert r["vout"][-1] == pytest.approx(24.0, abs=0.05)
    assert r["duty"].min() >= config.DUTY_MIN and r["duty"].max() <= config.DUTY_MAX
    assert r["il"].min() >= -1e-9


# ------------------------------------------------------------------ metrics
def test_metrics_on_first_order_step():
    t = np.linspace(0, 1, 10001)
    v = 24 * (1 - np.exp(-t / 0.1))
    assert M.rise_time(t, v, 24.0, v0=0.0) == pytest.approx(0.1 * np.log(9), rel=1e-2)
    assert M.settling_time(t, v, 24.0, tol=0.02) == pytest.approx(-0.1 * np.log(0.02), rel=1e-2)
    assert M.overshoot_pct(t, v, 24.0, v0=0.0) == 0.0
    assert M.steady_state_error(t, v, 24.0, 0.01) == pytest.approx(24 * np.exp(-10), abs=1e-3)


def test_overshoot_value():
    t = np.array([0, 1, 2, 3.0]); v = np.array([0, 26.4, 24, 24])
    assert M.overshoot_pct(t, v, 24.0, v0=0.0) == pytest.approx(10.0)


def test_undershoot_after_rise():
    t = np.arange(6.0); v = np.array([0, 22, 24, 19, 23, 24.0])
    assert M.undershoot_after_rise(t, v, 24.0) == pytest.approx(5.0)


def test_metrics_return_nan_for_empty_windows():
    t = np.array([0.0, 0.1, 0.2]); v = np.array([1.0, 2.0, 3.0])
    assert np.isnan(M.max_deviation(t, v, 24.0, t_start=1.0))
    assert np.isnan(M.rmse(t, v, 24.0, t_start=1.0))
    assert np.isnan(M.mae(t, v, 24.0, t_start=1.0))
    assert np.isnan(M.recovery_time(t, v, 24.0, t_dist=1.0))
