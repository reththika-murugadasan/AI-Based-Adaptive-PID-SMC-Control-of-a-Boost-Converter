"""Reusable time-response metrics (all computed from arrays, never by eye).

Conventions
* ``t`` [s], ``v`` output voltage [V], ``vref`` target [V] (scalar).
* Step metrics are measured on the window [t_start, t_end] of a step towards
  ``vref`` that starts from ``v0 = v(t_start)``.
"""
from __future__ import annotations

import numpy as np


def _window(t, *arrays, t_start=None, t_end=None):
    t = np.asarray(t, float)
    m = np.ones_like(t, bool)
    if t_start is not None: m &= t >= t_start
    if t_end is not None: m &= t <= t_end
    return (t[m],) + tuple(np.asarray(a, float)[m] for a in arrays)


def rise_time(t, v, vref, v0=None, lo=0.1, hi=0.9):
    """10%-90% rise time of a step from v0 to vref (nan if never reached)."""
    t, v = np.asarray(t, float), np.asarray(v, float)
    v0 = v[0] if v0 is None else v0
    span = vref - v0
    if span == 0:
        return float("nan")
    frac = (v - v0) / span
    i_lo = np.argmax(frac >= lo) if np.any(frac >= lo) else None
    i_hi = np.argmax(frac >= hi) if np.any(frac >= hi) else None
    if i_lo is None or i_hi is None:
        return float("nan")
    return float(t[i_hi] - t[i_lo])


def settling_time(t, v, vref, tol=0.02, t0=None):
    """Time after t0 at which |v - vref| <= tol*|vref| for good (nan if never)."""
    t, v = np.asarray(t, float), np.asarray(v, float)
    t0 = t[0] if t0 is None else t0
    if t[-1] < t0:
        return float("nan")                    # the run ended before t0
    outside = np.abs(v - vref) > tol * abs(vref)
    outside &= t >= t0
    if outside[-1]:
        return float("nan")
    idx = np.nonzero(outside)[0]
    return float(t[idx[-1] + 1] - t0) if idx.size else 0.0


def overshoot_pct(t, v, vref, v0=None):
    """Peak overshoot above vref [%] of the step size (0 if none)."""
    v = np.asarray(v, float)
    v0 = v[0] if v0 is None else v0
    span = vref - v0
    if span == 0:
        return float("nan")
    sg = np.sign(span)
    return float(max(0.0, np.max(sg * v) - sg * vref) / abs(span) * 100.0)


def steady_state_error(t, v, vref, window=0.01):
    """Mean (vref - v) over the last ``window`` seconds [V]."""
    t, v = np.asarray(t, float), np.asarray(v, float)
    m = t >= t[-1] - window
    return float(np.mean(vref - v[m]))


def rmse(t, v, vref, t_start=None, t_end=None):
    t, v = _window(t, v, t_start=t_start, t_end=t_end)
    if v.size == 0:
        return float("nan")                    # e.g. episode ended before t_start
    return float(np.sqrt(np.mean((np.asarray(vref) - v) ** 2)))


def mae(t, v, vref, t_start=None, t_end=None):
    t, v = _window(t, v, t_start=t_start, t_end=t_end)
    if v.size == 0:
        return float("nan")
    return float(np.mean(np.abs(np.asarray(vref) - v)))


def max_deviation(t, v, vref, t_start=None, t_end=None):
    """Largest |v - vref| in the window [V]."""
    t, v = _window(t, v, t_start=t_start, t_end=t_end)
    if v.size == 0:
        return float("nan")
    return float(np.max(np.abs(v - vref)))


def recovery_time(t, v, vref, t_dist, tol=0.02):
    """Time from a disturbance at t_dist until v stays within +-tol*vref."""
    return settling_time(t, v, vref, tol=tol, t0=t_dist)


def duty_variation(duty, t=None, t_start=None, t_end=None):
    """Mean |D[k]-D[k-1]| (smoothness of the control action)."""
    duty = np.asarray(duty, float)
    if t is not None:
        _, duty = _window(t, duty, t_start=t_start, t_end=t_end)
    return float(np.mean(np.abs(np.diff(duty)))) if duty.size > 1 else 0.0


def max_current(il):
    return float(np.max(np.asarray(il, float)))


def undershoot_after_rise(t, v, vref, frac=0.9):
    """Deepest dip below vref [V] after v first reaches frac*vref.

    Rise time/overshoot can look perfect while the response falls back
    afterwards; this exposes it.
    """
    t, v = np.asarray(t, float), np.asarray(v, float)
    if not np.any(v >= frac * vref):
        return float("nan")
    i = np.argmax(v >= frac * vref)
    return float(max(0.0, vref - v[i:].min()))
