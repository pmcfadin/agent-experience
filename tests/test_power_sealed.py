"""Sealed-study power analysis (analysis/power_sealed.py): variance recovery, type-I error, power."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location("power_sealed", Path(__file__).parent.parent / "analysis" / "power_sealed.py")
ps = importlib.util.module_from_spec(_spec)
sys.modules["power_sealed"] = ps
_spec.loader.exec_module(ps)

INTERCEPT = {"history": 0.4, "ticket": 0.5, "check": 0.4, "cell": 0.3}
SLOPE = {"history": 0.4, "ticket": 0.5, "check": 0.4, "cell": 0.4}
VC = ps.Components(INTERCEPT, SLOPE, baseline=0.5, checks_per_ticket=2.0)


def test_t_tail_matches_tables():
    p = ps.t_sf2(np.array([2.0, 2.776445, 1.0]), np.array([10, 4, 1]))
    assert p == pytest.approx([0.0734, 0.05, 0.5], abs=1e-3)
    assert ps.t_crit(4) == pytest.approx(2.7764, abs=1e-3)


def test_calibration_hits_marginal_rates():
    mu, beta = ps.calibrate(VC, 0.4, 0.10)
    sd = np.sqrt(sum(v ** 2 for v in INTERCEPT.values()) + sum(v ** 2 for v in SLOPE.values()) / 4)
    assert ps._marginal(mu, sd) == pytest.approx(0.4, abs=1e-6)
    assert ps._marginal(mu + beta, sd) == pytest.approx(0.5, abs=1e-6)


def test_simulator_recovers_variance_components():
    rng = np.random.default_rng(1)
    y_b, y_p, sizes = ps.simulate(VC, 10, 48, 16, rng, baseline=0.5, effect=0.0)
    assert y_b.mean() == pytest.approx(0.5, abs=0.03) and y_p.mean() == pytest.approx(0.5, abs=0.03)
    est = [ps.estimate_components(y_b[i], y_p[i], sizes) for i in range(len(y_b))]
    for kind, true in (("intercept", INTERCEPT), ("slope", SLOPE)):
        for level, sd in true.items():
            got = np.sqrt(np.mean([getattr(c, kind)[level] ** 2 for c in est]))
            assert got == pytest.approx(sd, abs=0.15), (kind, level)


def test_type_one_error_near_nominal():
    # no effect: the rejection rate stays near 0.05 (400 sims: a loose band)
    assert 0.02 <= ps.power(VC, 6, 24, sims=400, effect=0.0) <= 0.09


def test_power_rises_with_histories_and_tickets():
    p3, p10 = (ps.power(VC, h, 24, sims=300) for h in (3, 10))
    assert p10 > p3 + 0.1
    assert ps.power(VC, 10, 48, sims=300) > p10 - 0.05
    assert ps.power(VC, 10, 24, sims=300, effect=0.2) > p10


def test_primary_test_detects_a_clear_difference():
    rng = np.random.default_rng(0)
    d = (rng.random((8, 40)) < 0.7).astype(float) - (rng.random((8, 40)) < 0.2)
    res = ps.primary_test(d, np.full(20, 2))
    assert res["est"] == pytest.approx(d.mean()) and res["p"] < 0.001


def test_build_design_pairs_by_history_and_ticket():
    def ep(g, h, t, a, b):
        return {"group": g, "history": h, "ticket": t, "checks": {"lesson:x": a, "lesson:y": b}}
    eps = [ep(g, h, t, h == 0, t == "t1") for g in ("B", "P-all-ep", "A") for h in (0, 1) for t in ("t1", "t2")]
    eps.append(ep("B", 2, "t1", True, True))  # history 2 has no P-all-ep: left out
    d = ps.build_design(eps)
    assert d.y_new.shape == (2, 4) and list(d.sizes) == [2, 2] and np.array_equal(d.y_new, d.y_ref)
