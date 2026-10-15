"""Metrics for the effective-context programme.

Definitions (made explicit here so results are unambiguous):

Effective Context Capacity (ECC)
    Given a performance curve P(L) (accuracy vs. context length), a short-context
    baseline P0 = P(L_min), and a tolerance tau in (0, 1]:
        threshold = tau * P0
        ECC = the *first* length at which P(L) drops below `threshold`,
              linearly interpolated between the last length above and the first
              below. If P never drops below the threshold, ECC is right-censored
              at L_max.
    ECC is intentionally a *first-crossing* definition: it answers
    "up to what context length is performance still >= tau of baseline?"

Effective Context Mass (ECM)
    A threshold-free summary. Integral of P(L)/P0 over log-length, normalised
    to [0, 1]. ECM = 1.0 means no degradation across the measured range.
    ECM is more robust than a single threshold crossing because it uses the
    whole curve.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

try:  # scipy is available in this environment
    from scipy.optimize import curve_fit
except Exception:  # pragma: no cover
    curve_fit = None

# numpy 2.x renamed trapz -> trapezoid; support both.
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")


def _sorted_curve(lengths: Sequence[float], accs: Sequence[float]):
    pairs = sorted(zip(lengths, accs), key=lambda t: t[0])
    xs = np.asarray([p[0] for p in pairs], dtype=float)
    ys = np.asarray([p[1] for p in pairs], dtype=float)
    return xs, ys


def effective_context_capacity(
    lengths: Sequence[float], accs: Sequence[float], tau: float = 0.9
) -> Dict[str, float]:
    xs, ys = _sorted_curve(lengths, accs)
    if len(xs) < 2:
        raise ValueError("need at least two points")
    p0 = float(ys[0])
    threshold = tau * p0
    ecc = float(xs[-1])
    censored = True
    for i in range(1, len(xs)):
        if ys[i] < threshold:
            x0, y0, x1, y1 = xs[i - 1], ys[i - 1], xs[i], ys[i]
            if y0 == y1:
                ecc = float(x0)
            else:
                frac = (y0 - threshold) / (y0 - y1)
                ecc = float(x0 + frac * (x1 - x0))
            censored = False
            break
    return {
        "ecc": ecc,
        "tau": tau,
        "baseline_p0": p0,
        "threshold": threshold,
        "censored": float(censored),
        "max_length": float(xs[-1]),
    }


def effective_context_mass(
    lengths: Sequence[float], accs: Sequence[float]
) -> Dict[str, float]:
    xs, ys = _sorted_curve(lengths, accs)
    if len(xs) < 2:
        raise ValueError("need at least two points")
    p0 = float(ys[0])
    if p0 <= 0:
        return {"ecm": 0.0, "baseline_p0": p0}
    logx = np.log2(xs)
    norm = ys / p0
    area = _trapz(norm, logx)
    span = float(logx[-1] - logx[0])
    ecm = float(area / span) if span > 0 else float(norm[0])
    return {"ecm": ecm, "baseline_p0": p0, "log_span": span}


def _stretched_exp(x, p0, scale, beta):
    return p0 * np.exp(-((x / scale) ** beta))


def fit_stretched_exponential(
    lengths: Sequence[float], accs: Sequence[float]
) -> Dict[str, float]:
    """Fit P(L) = P0 * exp(-(L/scale)^beta). Returns params + R^2 (may fail)."""
    if curve_fit is None:
        return {"ok": 0.0, "reason": "scipy unavailable"}
    xs, ys = _sorted_curve(lengths, accs)
    if len(xs) < 4:
        return {"ok": 0.0, "reason": "need >=4 points"}
    try:
        p0_0 = float(ys[0])
        popt, _ = curve_fit(
            _stretched_exp,
            xs,
            ys,
            p0=[p0_0, float(xs[-1]), 1.0],
            bounds=([0.0, 1e-6, 0.05], [1.5, 1e9, 5.0]),
            maxfev=20000,
        )
        pred = _stretched_exp(xs, *popt)
        ss_res = float(np.sum((ys - pred) ** 2))
        ss_tot = float(np.sum((ys - np.mean(ys)) ** 2)) or 1e-12
        return {
            "ok": 1.0,
            "p0": float(popt[0]),
            "scale": float(popt[1]),
            "beta": float(popt[2]),
            "r2": 1.0 - ss_res / ss_tot,
        }
    except Exception as e:  # pragma: no cover
        return {"ok": 0.0, "reason": str(e)}
