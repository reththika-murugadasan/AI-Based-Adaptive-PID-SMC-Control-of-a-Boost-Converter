"""Reward-function tests (BoostConverterEnv)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from environment.boost_env import (REWARD_TERMS, BoostConverterEnv, duty_to_action,
                                   reward_terms)

OPT = dict(vin=12.0, load=20.0, vref=24.0, init="equilibrium", disturbance=None)


def total(error, d_error=0.0, duty=0.5, prev=0.5, il_max=3.0):
    return sum(reward_terms(error, d_error, duty, prev, il_max).values())


# --------------------------------------------------- 1. error decreases -> reward up
def test_reward_increases_as_error_shrinks():
    errs = [24, 10, 5, 2, 1, 0.5, 0.2, 0.1, 0.05, 0.0]
    r = [total(e) for e in errs]
    assert all(b > a for a, b in zip(r, r[1:]))


def test_small_offsets_are_still_clearly_penalised():
    """Precision matters: 0.2 V and 0.5 V offsets must cost real reward."""
    assert total(0.0) - total(0.2) > 0.3
    assert total(0.2) - total(2.0) > 0.4
    assert total(0.05) > total(0.2) > total(0.5)


def test_overshoot_costs_more_than_equal_undershoot():
    assert total(-2.0) < total(2.0)          # e<0 means Vout above Vref


def test_env_reward_improves_when_voltage_gets_closer():
    """Same state except the output voltage: the closer one earns more."""
    rs = []
    for v in (18.0, 21.0, 23.5, 24.0):
        e = BoostConverterEnv(); e.reset(seed=0, options=OPT)
        il, _ = e.converter.steady_state(0.5)
        e.converter.reset(IL=il, Vout=v); e.prev_duty = 0.5; e.prev_error = e.converter.Vref - v
        rs.append(e.step(duty_to_action(0.5))[1])
    assert all(b > a for a, b in zip(rs, rs[1:]))


# ------------------------------------------- 2. large persistent error penalised
def test_persistent_error_accumulates_a_large_penalty():
    """Holding the output 2 V away for 100 steps must cost far more than regulating."""
    good = sum(total(0.0) for _ in range(100))
    offset = sum(total(2.0) for _ in range(100))
    far = sum(total(8.0) for _ in range(100))
    assert good - offset > 50 and offset - far > 30


def test_starving_the_converter_scores_worse_than_regulating():
    def episode(duty):
        e = BoostConverterEnv(max_steps=150, disturbance_prob=0.0)
        e.reset(seed=0, options=OPT)
        total_r = 0.0
        for _ in range(150):
            _, r, term, trunc, _ = e.step(duty_to_action(duty))
            total_r += r
            if term or trunc:
                break
        return total_r
    assert episode(0.5) > episode(0.05) + 30     # ideal duty vs starved (V collapses)


# ------------------------------------------------ 3. large duty changes penalised
def test_duty_change_penalty_monotonic_and_bounded():
    pen = [reward_terms(0, 0, 0.5 + dd, 0.5, 3)["duty"] for dd in (0, 0.01, 0.05, 0.2, 0.5, 0.9)]
    assert pen[0] == pytest.approx(0.0)
    assert all(b < a for a, b in zip(pen, pen[1:]))
    assert pen[-1] > -1.0                          # bounded: legitimate big moves are affordable


def test_small_jitter_is_not_free():
    assert reward_terms(0, 0, 0.54, 0.5, 3)["duty"] < -0.01


def test_env_penalises_a_duty_jump():
    rs = []
    for duty in (0.5, 0.9):
        e = BoostConverterEnv(); e.reset(seed=0, options=OPT)
        e.converter.reset(IL=2.4, Vout=24.0); e.prev_duty = 0.5; e.prev_error = 0.0
        rs.append(e.step(duty_to_action(duty))[4]["reward_terms"]["duty"])
    assert rs[0] == pytest.approx(0.0, abs=1e-6) and rs[1] < -0.1


def test_alternating_duty_scores_worse_than_steady_duty():
    """A limit cycle (D flipping 0.4/0.6) must lose to the same mean duty held."""
    def run(seq):
        e = BoostConverterEnv(max_steps=len(seq), disturbance_prob=0.0)
        e.reset(seed=0, options=OPT)
        il_eq, v_eq = e.converter.steady_state(0.5)
        e.converter.reset(IL=il_eq, Vout=v_eq)
        return sum(e.step(duty_to_action(d))[1] for d in seq)
    n = 60
    assert run([0.5] * n) > run([0.4, 0.6] * (n // 2))


# --------------------------------------------- 4. unsafe current / state penalised
def test_soft_current_penalty_monotonic():
    pen = [reward_terms(0, 0, .5, .5, i)["cur"] for i in (5, 12, 14, 18, 25, 30)]
    assert pen[0] == pen[1] == 0.0
    assert all(b < a for a, b in zip(pen[1:], pen[2:]))


def test_normal_inrush_is_not_penalised():
    """The ~9 A cold-start inrush of the PID/SMC stays below the soft limit."""
    assert reward_terms(0, 0, .5, .5, 9.5)["cur"] == 0.0


def test_over_current_trip_terminates_with_penalty():
    e = BoostConverterEnv(disturbance_prob=0.0); e.reset(seed=0, options=OPT)
    e.converter.IL = config.ENV_I_MAX + 5.0
    _, r, term, trunc, info = e.step(duty_to_action(0.5))
    assert term and "over-current" in info["failure"] and r == -config.ENV_REWARD["fail_penalty"]


def test_over_voltage_trip_terminates_with_penalty():
    e = BoostConverterEnv(disturbance_prob=0.0); e.reset(seed=0, options=OPT)
    e.converter.Vout, e.converter.IL = config.ENV_V_MAX + 5.0, 1.0
    _, r, term, trunc, info = e.step(duty_to_action(0.5))
    assert term and r == -config.ENV_REWARD["fail_penalty"]


def test_invalid_action_is_penalised():
    e = BoostConverterEnv(disturbance_prob=0.0); e.reset(seed=0, options=OPT)
    il_eq, v_eq = e.converter.steady_state(0.5)
    e.converter.reset(IL=il_eq, Vout=v_eq)
    e.prev_duty = 0.5
    r_ok = e.step(duty_to_action(0.5))[1]
    e.reset(seed=0, options=OPT); e.converter.reset(IL=2.4, Vout=24.0); e.prev_duty = 0.5
    r_bad = e.step(np.array([np.nan], np.float32))[1]
    assert r_bad < r_ok - 0.5


def test_tripping_is_never_better_than_surviving():
    """Even a very bad surviving step (cold start, 28 V ref) beats the trip penalty
    by an order of magnitude, and the penalty is not wildly out of scale."""
    worst_normal = sum(reward_terms(28.0, 5.0, 0.95, 0.05, 20.0).values())
    assert worst_normal > -10
    assert 40 <= config.ENV_REWARD["fail_penalty"] <= 100


# ------------------------------------------------------ 5. finite and bounded
def test_all_terms_present_and_finite_over_a_wide_grid():
    for e in np.linspace(-40, 40, 41):
        for dd in (0, 0.3, 0.9):
            for il in (0, 5, 25):
                t = reward_terms(e, e * 0.1, 0.5 + dd / 2, 0.5, il)
                assert set(t) == set(REWARD_TERMS)
                assert all(np.isfinite(v) for v in t.values())
                assert -12 < sum(t.values()) < 1.6


def test_regulated_state_has_positive_bounded_reward():
    r = total(0.0)
    assert 1.0 < r <= 1.5 + 1e-9


def test_normal_rollouts_have_finite_bounded_rewards():
    e = BoostConverterEnv(component_tolerance=0.1)
    rng = np.random.default_rng(0)
    for ep in range(4):
        e.reset(seed=ep)
        done = False
        while not done:
            _, r, term, trunc, info = e.step(rng.uniform(-1, 1, size=1))
            assert np.isfinite(r) and -config.ENV_REWARD["fail_penalty"] - 1 <= r <= 1.6
            assert set(info["reward_terms"]) == set(REWARD_TERMS)
            done = term or trunc


def test_episode_stats_terms_sum_to_return():
    e = BoostConverterEnv(max_steps=50, disturbance_prob=0.0)
    e.reset(seed=0, options=OPT)
    for _ in range(50):
        _, r, term, trunc, info = e.step(duty_to_action(0.5))
        if term or trunc:
            break
    st = info["episode_stats"]
    assert st["return_"] == pytest.approx(sum(st["terms"].values()), abs=1e-6)
