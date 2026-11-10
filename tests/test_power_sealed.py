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
    assert d.y_new.shape == (1, 2, 4) and list(d.sizes) == [2, 2] and np.array_equal(d.y_new, d.y_ref)


NOISY = ps.Components({"history": 0.3, "ticket": 0.3, "check": 0.3, "cell": 0.0},
                      {"history": 0.3, "ticket": 0.3, "check": 0.3, "cell": 0.3}, baseline=0.5, run=1.0)


def test_replicates_split_cell_and_episode_variance():
    rng = np.random.default_rng(2)
    y_b, y_p, sizes = ps.simulate(NOISY, 10, 48, 12, rng, baseline=0.5, effect=0.0, runs=3)
    assert y_b.shape == (12, 3, 10, 96)
    est = [ps.estimate_components(y_b[i], y_p[i], sizes) for i in range(12)]
    assert np.sqrt(np.mean([c.run ** 2 for c in est])) == pytest.approx(1.0, abs=0.25)
    assert np.sqrt(np.mean([c.slope["cell"] ** 2 for c in est])) == pytest.approx(0.3, abs=0.2)
    # one run per cell cannot tell them apart: everything is reported as cell variance, run = 0
    one = ps.estimate_components(y_b[0, :1], y_p[0, :1], sizes)
    assert one.run == 0 and one.slope["cell"] > 0.5


def test_one_run_reproduces_the_single_run_results():
    # pinned from the version without replicates (same seeds, same random stream)
    assert ps.power(VC, 6, 24, runs=1, sims=300) == pytest.approx(0.28667, abs=1e-4)
    assert ps.power(VC, 4, 36, sims=300, seed=3) == pytest.approx(0.22333, abs=1e-4)


def test_power_rises_with_runs_when_episode_noise_dominates():
    p1, p3 = (ps.power(NOISY, 6, 24, r, sims=400) for r in (1, 3))
    assert p3 > p1 + 0.1
    assert 0.02 <= ps.power(NOISY, 6, 24, 3, sims=400, effect=0.0) <= 0.09  # type I stays nominal


def test_build_design_keeps_replicate_runs():
    def ep(g, h, t, v):
        return {"group": g, "history": h, "ticket": t, "checks": {"lesson:x": v}}
    eps = [ep(g, h, t, r == 0) for g in ("B", "P-all-ep") for h in (0, 1) for t in ("t1", "t2") for r in range(3)]
    d = ps.build_design(eps)
    assert d.y_new.shape == (3, 2, 2) and d.dropped == 0
    assert ps.cell_diff(d.y_new, d.y_ref).shape == (2, 2)
