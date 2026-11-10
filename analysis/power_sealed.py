"""Power analysis for the sealed study's primary test (SPEC §13 "Roadmap to the sealed study",
docs/design/house-lib-core-grid.md §10): B vs P-all-ep at L64-c50, lesson-check pass on kept lessons,
paired by ticket.

Design. The unit of observation is a lesson check (history h, eval ticket t, check k of that ticket).
Every history runs every ticket in both groups (P-all-ep is built from the same history's B bank), so
the pairs are (h, t, k) cells and the groups differ only in the treatment.

Model (simulated; logit scale). Pass probability of group g in a cell:

    logit p = mu + beta [g = B] + a_h + a_t + a_k + a_ht  +  [g = B] (b_h + b_t + b_k + b_ht)

with independent normal effects for history (h), ticket (t), check within ticket (k, shared by every
history) and the ticket-by-history cell (ht, shared by a ticket's checks). The a-terms are random
intercepts (they cancel in a paired contrast but set the Bernoulli noise level); the b-terms are the
group-by-cluster slopes, which are what limits power: more histories average out b_h and b_ht, more
eval tickets average out b_t and b_k. (The B and P-all-ep slopes are split +-1/2 between groups; only
their sum matters for the contrast.) mu and beta are solved so that the marginal pass rates are the
baseline (P-all-ep) and baseline + 10 points.

Replicates. With R runs per (group, history, ticket) (4.3 at the primary cell), each episode also gets an
episode effect: one normal draw (SD `run`) shared by its checks, independent across runs and groups. It is
the run-to-run noise that a ticket-by-history slope cannot be told apart from when R = 1. R enters the
primary test only through the cell mean: z_ht uses the difference of the two groups' run-averaged check
outcomes, so MS_e shrinks as the episode effect averages over R runs, while history and ticket variances
and the true ticket x history slope do not shrink. The test function is the same for every R.

Primary test (what is preregistered). Not the mixed logistic fit itself: a fitted GLMM with ~10 clusters
is slow, fails to converge often and has no exact small-sample df. Instead a linear mixed model on the
paired contrast, by crossed-random-effects ANOVA, which is cheap, closed-form and respects both
clusterings:

    z_ht = (n_t / mean n) * mean_k (y_B - y_P)       one number per (history, ticket) cell
    z_ht = delta + u_h + v_t + e_ht                   histories and tickets crossed, one observation per cell
    delta_hat = mean z  (= the pooled lesson-check pass difference)
    Var(delta_hat) = (MS_h + MS_t - MS_e) / (H T), floored at MS_e / (H T), Satterthwaite df, t test

The n_t weight makes delta_hat the pooled difference over checks (tickets with more kept-lesson checks
count more, as in the pass rate), and per-ticket check counts are handled without a nested model.
Ticket and history effects of the contrast and the cell noise (including Bernoulli noise) are
all inside the three mean squares. The null distribution is checked by simulation (`power(..., effect=0)`, and
tests/test_power_sealed.py).

Variance components come from stage 4.3 data (`estimate_components`): method of moments on pair
products of the check-level mean (intercept components) and difference (slope components) of the two
groups, unbiased without centring (the product of two cells that share nothing estimates mu^2), floored
at 0, then divided by (p(1-p))^2 to move from the pass-rate scale to the logit scale (delta method:
trusted for logit SDs up to about 1). With 3 histories the history components are very uncertain, so
the table is a plan, to be re-run with the final 4.3 data. With R = 1 the episode effect cannot be
separated from the ticket x history slope and all of it is reported as `cell` (the conservative case:
replicates then look useless); with R > 1, check pairs of one cell from different runs give `cell` and the
excess covariance of pairs from the same run gives `run`. Without replicates the report shows both
extremes (all `cell`, all `run`) so the plan can be read as a range.

    uv run python analysis/power_sealed.py --from-batches exp22-grid-L64-c50-w1 exp23-grid-static-L64-c50-w1 \
        --world L64-c50-w1 [--runs-dir runs] [--baseline 0.45] [--sims 2000]   # power over 1-3 runs per cell too
    uv run python analysis/power_sealed.py --variances history=0.3,ticket=0.5,check=0.3,cell=0.3 \
        --slope-variances history=0.3,ticket=0.5,check=0.3,cell=0.3 --baseline 0.4
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sqlite3
from math import lgamma
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from elh.tasks.families.house_lib.grid import kept_lessons, scored_checks

LEVELS = ("history", "ticket", "check", "cell")  # cell = ticket x history
EFFECT = 0.10  # smallest effect of interest: B - P-all-ep, percentage points as a fraction
ALPHA, TARGET = 0.05, 0.80
HISTORIES, TICKETS, RUNS, MAX_HISTORIES = range(3, 11), (24, 36, 48), (1, 2, 3), 10
GROUPS = ("B", "P-all-ep")
_GH_X, _GH_W = np.polynomial.hermite_e.hermegauss(80)
_GH_W = _GH_W / _GH_W.sum()


# ---- numerics (numpy only: Student t tail, bisection) ----

def _betainc(a: np.ndarray, b: float, x: np.ndarray) -> np.ndarray:
    """Regularised incomplete beta I_x(a, b), x in [0, 1] (continued fraction, Numerical Recipes)."""
    a, x = np.broadcast_arrays(np.asarray(a, dtype=float), np.asarray(x, dtype=float))
    swap = x > (a + 1) / (a + b + 2)
    aa, bb, xx = np.where(swap, b, a), np.where(swap, a, b), np.where(swap, 1 - x, x)
    tiny = 1e-300
    fix = lambda v: np.where(np.abs(v) < tiny, tiny, v)  # noqa: E731
    c, d = np.ones_like(xx), 1 / fix(1 - (aa + bb) * xx / (aa + 1))
    h = d
    for m in range(1, 400):
        for num in (m * (bb - m) * xx / ((aa + 2 * m - 1) * (aa + 2 * m)),
                    -(aa + m) * (aa + bb + m) * xx / ((aa + 2 * m) * (aa + 2 * m + 1))):
            d, c = 1 / fix(1 + num * d), fix(1 + num / c)
            h = h * d * c
        if np.all(np.abs(d * c - 1) < 1e-12):
            break
    lg = np.vectorize(lgamma)
    with np.errstate(divide="ignore", invalid="ignore"):
        front = np.exp(lg(aa + bb) - lg(aa) - lg(bb) + aa * np.log(xx) + bb * np.log1p(-xx)) / aa
    val = np.where(xx <= 0, 0.0, np.where(xx >= 1, 1.0, front * h))
    return np.where(swap, 1 - val, val)


def t_sf2(t: np.ndarray, df: np.ndarray) -> np.ndarray:
    """Two-sided tail probability P(|T| > |t|) of Student t with df degrees of freedom."""
    t, df = np.broadcast_arrays(np.abs(np.asarray(t, dtype=float)), np.asarray(df, dtype=float))
    return _betainc(df / 2, 0.5, df / (df + t * t))


def bisect(f, lo: float, hi: float, tol: float = 1e-10) -> float:
    """Root of an increasing function f on [lo, hi]."""
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
        if hi - lo < tol:
            break
    return (lo + hi) / 2


def t_crit(df: float, alpha: float = ALPHA) -> float:
    """Two-sided critical value of Student t."""
    return bisect(lambda x: alpha - float(t_sf2(x, df)), 0.0, 1e4, 1e-8)


@dataclass(frozen=True)
class Components:
    """Logit-scale standard deviations per level: `intercept` (both groups) and `slope` (B's departure
    from P-all-ep), plus the design they were measured on."""
    intercept: dict = field(default_factory=lambda: dict.fromkeys(LEVELS, 0.0))
    slope: dict = field(default_factory=lambda: dict.fromkeys(LEVELS, 0.0))
    baseline: float = 0.4  # P-all-ep marginal pass rate
    checks_per_ticket: float = 2.0
    observed_effect: float | None = None
    run: float = 0.0  # SD of the episode effect: one draw per run, shared by its checks, independent across runs and groups
    runs: int = 1  # runs per (group, history, ticket) in the data the SDs came from

    def sd(self, kind: str, level: str) -> float:
        return float((self.intercept if kind == "intercept" else self.slope)[level])


# ---- design from episodes ----

@dataclass
class Design:
    """Paired lesson-check outcomes: y_new, y_ref are (runs, histories, checks) 0/1 arrays; checks are
    grouped by ticket (`sizes[t]` consecutive checks)."""
    y_new: np.ndarray
    y_ref: np.ndarray
    sizes: np.ndarray
    dropped: int = 0


def load_checks(runs_dir: Path, batches: list[str], world: str) -> list[dict]:
    """Kept-lesson check outcomes of every checkpoint episode in the batches (database opened read-only)."""
    kept, out = kept_lessons(world), []
    db = sqlite3.connect(f"file:{Path(runs_dir).resolve() / 'elh.sqlite3'}?mode=ro", uri=True)
    for name in batches:
        for g, h, task, v in db.execute(
                "SELECT test_group, history, task_id, verdict_json FROM episodes "
                "WHERE batch_id LIKE ? AND split = 'checkpoint' ORDER BY started_at, episode_id", (f"{name}/%",)):
            sc = scored_checks(json.loads(v)["checks"], kept)
            out.append({"group": g, "history": h, "ticket": task,
                        "checks": {k: bool(x) for k, x in sc.items() if k.startswith("lesson:")}})
    return out


def build_design(eps: list[dict], a: str = GROUPS[0], b: str = GROUPS[1]) -> Design:
    """Pair group a with b by (history, ticket) and check; keeps histories with both groups and tickets
    present in all of them (anything else is counted in `dropped`)."""
    runs: dict[tuple, list[dict]] = {}
    for e in eps:
        if e["group"] in (a, b):
            runs.setdefault((e["group"], e["history"], e["ticket"]), []).append(e["checks"])
    R = min(len(v) for v in runs.values())  # replicate runs (in start order) per cell; extras are dropped
    cells = {k: v[:R] for k, v in runs.items()}
    hists = sorted({h for g, h, _ in cells if g == a} & {h for g, h, _ in cells if g == b})
    if not hists:
        raise ValueError(f"no history has both {a} and {b} episodes")
    tickets = set.intersection(*({t for g2, h2, t in cells if g2 == g and h2 == h} for g in (a, b) for h in hists))
    tickets = sorted(tickets)
    names = {t: sorted(set.intersection(*(set(c) for g in (a, b) for h in hists for c in cells[g, h, t]))) for t in tickets}
    tickets = [t for t in tickets if names[t]]
    cols = [(t, k) for t in tickets for k in names[t]]
    ya = np.array([[[cells[a, h, t][r][k] for t, k in cols] for h in hists] for r in range(R)], dtype=float)
    yb = np.array([[[cells[b, h, t][r][k] for t, k in cols] for h in hists] for r in range(R)], dtype=float)
    dropped = sum(len(v) for v in runs.values()) - 2 * R * len(hists) * len(tickets)
    return Design(ya, yb, np.array([len(names[t]) for t in tickets]), dropped)


# ---- primary test ----

def cell_diff(y_new: np.ndarray, y_ref: np.ndarray) -> np.ndarray:
    """B - P-all-ep per (history, check), averaged over the R runs of each cell: inputs (..., R, H, N)."""
    return (np.asarray(y_new, dtype=float) - np.asarray(y_ref, dtype=float)).mean(-3)


def primary_test(d: np.ndarray, sizes: np.ndarray, alpha: float = ALPHA) -> dict:
    """Crossed-random-effects ANOVA test of mean(d) = 0; d = y_B - y_P with shape (..., H, N) whose last
    axis is the checks, ticket by ticket (`sizes`). Returns arrays est, se, df, t, p, reject over the leading axes."""
    sizes = np.asarray(sizes)
    T, H = len(sizes), d.shape[-2]
    if H < 2 or T < 2:
        raise ValueError("need at least 2 histories and 2 tickets")
    z = np.add.reduceat(d, np.r_[0, np.cumsum(sizes)[:-1]], axis=-1) / sizes.mean()  # (..., H, T)
    est = z.mean((-1, -2))
    rows, cols = z.mean(-1), z.mean(-2)
    ms_h = T * ((rows - est[..., None]) ** 2).sum(-1) / (H - 1)
    ms_t = H * ((cols - est[..., None]) ** 2).sum(-1) / (T - 1)
    resid = z - rows[..., :, None] - cols[..., None, :] + est[..., None, None]
    ms_e = (resid ** 2).sum((-1, -2)) / ((H - 1) * (T - 1))
    var = (ms_h + ms_t - ms_e) / (H * T)
    floor = ms_e / (H * T)
    use_floor = var <= floor
    var = np.where(use_floor, floor, var)
    c = np.stack([ms_h, ms_t, -ms_e]) / (H * T)
    dfs = np.array([H - 1, T - 1, (H - 1) * (T - 1)], dtype=float).reshape(3, *([1] * est.ndim))
    with np.errstate(divide="ignore", invalid="ignore"):
        df = np.where(use_floor, (H - 1) * (T - 1), var ** 2 / (c ** 2 / dfs).sum(0))
    df = np.clip(np.nan_to_num(df, nan=1.0), 1.0, None)
    se = np.sqrt(var)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, est / se, 0.0)
    p = t_sf2(t, df)
    return {"est": est, "se": se, "df": df, "t": t, "p": p, "reject": p < alpha}


# ---- variance components ----

def estimate_components(y_new: np.ndarray, y_ref: np.ndarray, sizes: np.ndarray) -> Components:
    """Method-of-moments logit-scale SDs from paired check outcomes of shape (runs, histories, checks)
    (module docstring). With one run per cell the episode effect cannot be told from the ticket x history
    slope, so all of it is reported as `cell` (the conservative case, `run` = 0). With R > 1 runs, pairs of
    checks of one cell from different runs give `cell`, and the excess covariance of pairs from the same
    run gives the episode effect `run`."""
    R, H, N = y_new.shape
    t_of = np.repeat(np.arange(len(sizes)), sizes)
    rr = np.repeat(np.arange(R), H * N)
    hh, tt, kk = (np.tile(np.repeat(np.arange(H), N), R), np.tile(np.tile(t_of, H), R),
                  np.tile(np.tile(np.arange(N), H), R))
    p = float((y_new.mean() + y_ref.mean()) / 2)
    scale = max(p * (1 - p), 1e-6)
    iu = np.triu_indices(R * H * N, 1)
    sr, sh, st, sk = ((v[:, None] == v)[iu] for v in (rr, hh, tt, kk))

    def comps(x: np.ndarray) -> tuple[dict, float]:
        prod = np.outer(x, x)[iu]
        mean = lambda m: prod[m].mean() if m.any() else float("nan")  # noqa: E731
        base, a_h = mean(~sh & ~st), mean(sh & ~st)
        a_t, a_k = mean(~sh & st & ~sk), mean(~sh & st & sk)
        same, other = mean(sh & st & ~sk & sr), mean(sh & st & ~sk & ~sr)
        cell, run = (same, 0.0) if R == 1 else (other, same - other)
        raw = {"history": a_h - base, "ticket": a_t - base, "check": a_k - a_t, "cell": cell - a_h - a_t + base}
        sd = lambda v: math.sqrt(max(0.0, 0.0 if math.isnan(v) else v)) / scale  # noqa: E731
        return {k: sd(v) for k, v in raw.items()}, sd(run)

    mean_y, diff = ((y_new + y_ref) / 2).ravel(), (y_new - y_ref).ravel()
    slope, run = comps(diff)
    # the difference of two independent episode effects has variance 2 sd^2
    return Components(comps(mean_y)[0], slope, baseline=float(y_ref.mean()),
                      checks_per_ticket=float(np.mean(sizes)), observed_effect=float((y_new - y_ref).mean()),
                      run=run / math.sqrt(2), runs=R)


# ---- simulation ----

def _marginal(mu: float, sd: float) -> float:
    return float((_GH_W / (1 + np.exp(-(mu + sd * _GH_X)))).sum())


def calibrate(vc: Components, baseline: float, effect: float) -> tuple[float, float]:
    """(mu, beta) so the marginal pass rates are baseline (P-all-ep) and baseline + effect (B)."""
    sd = math.sqrt(sum(vc.sd("intercept", k) ** 2 + vc.sd("slope", k) ** 2 / 4 for k in LEVELS) + vc.run ** 2)
    mu = bisect(lambda m: _marginal(m, sd) - baseline, -30, 30)
    hi = bisect(lambda m: _marginal(m, sd) - (baseline + effect), -30, 30)
    return mu, hi - mu


def ticket_sizes(tickets: int, checks_per_ticket: float) -> np.ndarray:
    """Checks per ticket: floor / ceil of the mean, mixed to hit it."""
    cum = np.floor(np.arange(1, tickets + 1) * checks_per_ticket + 0.5)
    return np.diff(np.r_[0, cum]).astype(int)


def simulate(vc: Components, histories: int, tickets: int, sims: int, rng: np.random.Generator, *,
             baseline: float, effect: float, runs: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(y_B, y_P, sizes) of `sims` datasets, each y of shape (sims, runs, histories, checks): `runs`
    independent episodes per group, history and ticket."""
    sizes = ticket_sizes(tickets, vc.checks_per_ticket)
    N, t_of = sizes.sum(), np.repeat(np.arange(tickets), sizes)
    mu, beta = calibrate(vc, baseline, effect)

    def draw(kind: str) -> np.ndarray:
        sd = lambda k: vc.sd(kind, k)  # noqa: E731
        eff = rng.normal(size=(sims, histories, 1)) * sd("history")
        eff = eff + rng.normal(size=(sims, 1, tickets))[..., t_of] * sd("ticket")
        eff = eff + rng.normal(size=(sims, 1, N)) * sd("check")
        return eff + rng.normal(size=(sims, histories, tickets))[..., t_of] * sd("cell")

    a, b = draw("intercept")[:, None], draw("slope")[:, None]
    shape = (sims, runs, histories, tickets)
    e_p, e_b = ((rng.normal(size=shape)[..., t_of] * vc.run) if vc.run else 0.0 for _ in range(2))
    expit = lambda x: 1 / (1 + np.exp(-x))  # noqa: E731
    y_p = rng.random((sims, runs, histories, N)) < expit(mu + a - b / 2 + e_p)
    y_b = rng.random((sims, runs, histories, N)) < expit(mu + beta + a + b / 2 + e_b)
    return y_b.astype(float), y_p.astype(float), sizes


def power(vc: Components, histories: int, tickets: int, runs: int = 1, *, sims: int = 2000, seed: int = 0,
          baseline: float | None = None, effect: float = EFFECT, alpha: float = ALPHA) -> float:
    """Rejection rate of the primary test at the given effect (0 gives the type-I error)."""
    rng = np.random.default_rng([seed, histories, tickets] + ([runs] if runs > 1 else []))
    y_b, y_p, sizes = simulate(vc, histories, tickets, sims, rng, runs=runs,
                               baseline=vc.baseline if baseline is None else baseline, effect=effect)
    return float(primary_test(cell_diff(y_b, y_p), sizes, alpha)["reject"].mean())


def power_table(vc: Components, *, histories=HISTORIES, tickets=TICKETS, runs=RUNS, **kw) -> dict[tuple, dict[int, float]]:
    """Power by (runs, eval tickets) row and histories column."""
    return {(r, t): {h: power(vc, h, t, r, **kw) for h in histories} for r in runs for t in tickets}


def smallest(table: dict[tuple, dict[int, float]], target: float = TARGET) -> dict[tuple, int | None]:
    """Fewest histories reaching the target, per (runs, tickets) row (None: not within the table)."""
    return {t: next((h for h, p in sorted(row.items()) if p >= target), None) for t, row in table.items()}


# ---- report ----

def parse_sds(text: str) -> dict:
    out = dict.fromkeys(LEVELS, 0.0)
    for part in filter(None, text.split(",")):
        k, v = part.split("=")
        if k not in out:
            raise SystemExit(f"unknown level {k!r}; use {', '.join(LEVELS)}")
        out[k] = float(v)
    return out


def extreme_splits(vc: Components) -> dict[str, Components]:
    """Without replicates the slope `cell` SD mixes episode noise with true ticket x history variance.
    The two extremes: all of it true cell variance (the default, conservative) or all of it episode noise
    (a cell slope s equals the difference of two independent episode effects of SD s / sqrt 2)."""
    noise = dataclasses.replace(vc, slope={**vc.slope, "cell": 0.0}, run=vc.slope["cell"] / math.sqrt(2))
    return {"all ticket x history (conservative)": vc, "all episode noise": noise}


def report(vc: Components, *, sims: int, seed: int, label: str, effect: float = EFFECT, runs=RUNS) -> None:
    print(f"## Power of the primary test, B vs P-all-ep ({label})\n")
    print(f"Smallest effect {effect * 100:.0f} points, baseline (P-all-ep) {vc.baseline:.2f}, "
          f"{vc.checks_per_ticket:.2f} checks per ticket, alpha {ALPHA}, {sims} simulations per cell; "
          f"the SDs come from {vc.runs} run(s) per cell.")
    if vc.observed_effect is not None:
        print(f"Observed B - P-all-ep in the data: {vc.observed_effect * 100:+.1f} points (not used).")
    cases = extreme_splits(vc) if vc.runs == 1 and vc.run == 0 else {"as estimated": vc}
    for name, c in cases.items():
        print(f"\n### {name}\n\nLogit-scale SDs (`intercept` | `slope` = B's departure from P-all-ep); "
              f"episode effect `run` = {c.run:.2f}:\n")
        print("| Level | Intercept | Slope |\n|---|---|---|")
        for k in LEVELS:
            print(f"| {k} | {c.sd('intercept', k):.2f} | {c.sd('slope', k):.2f} |")
        table = power_table(c, sims=sims, seed=seed, effect=effect, runs=runs)
        print("\n| Runs per cell | Eval tickets | " + " | ".join(f"{h} hist." for h in HISTORIES) + " |")
        print("|---|---|" + "---|" * len(HISTORIES))
        for (r, t), row in table.items():
            print(f"| {r} | {t} | " + " | ".join(f"{p:.2f}" for p in row.values()) + " |")
        print()
        for (r, t), h in smallest(table).items():
            print(f"- {r} run(s), {t} tickets: " + (f"power >= {TARGET} from {h} histories" if h else
                                                      f"power < {TARGET} even at {MAX_HISTORIES} histories"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--from-batches", nargs=2, metavar=("B_BATCH", "P_BATCH"),
                    help="batch prefixes holding group B and group P-all-ep (variances from their episodes)")
    ap.add_argument("--world", help="world id for the kept lessons, e.g. L64-c50-w1")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--variances", help="manual intercept SDs, e.g. history=0.3,ticket=0.5,check=0.3,cell=0.3")
    ap.add_argument("--slope-variances", default="", help="manual group-slope SDs, same keys")
    ap.add_argument("--run-sd", type=float, default=0.0, help="manual episode-effect SD (needs replicates to estimate)")
    ap.add_argument("--baseline", type=float, help="P-all-ep pass rate (default: the data's)")
    ap.add_argument("--checks-per-ticket", type=float)
    ap.add_argument("--sims", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    if bool(args.from_batches) == bool(args.variances):
        ap.error("give --from-batches (with --world) or --variances")
    if args.from_batches:
        if not args.world:
            ap.error("--from-batches needs --world")
        eps = load_checks(Path(args.runs_dir), list(args.from_batches), args.world)
        design = build_design(eps)
        vc = estimate_components(design.y_new, design.y_ref, design.sizes)
        res = {k: float(v) for k, v in primary_test(cell_diff(design.y_new, design.y_ref), design.sizes).items()}
        half = t_crit(res['df']) * res['se']
        print(f"Data: {design.y_new.shape[1]} histories x {len(design.sizes)} tickets x {vc.runs} run(s) x "
              f"{vc.checks_per_ticket:.2f} checks, P-all-ep pass {vc.baseline:.2f}, dropped {design.dropped} episodes. "
              f"Primary test on it: B - P-all-ep {res['est'] * 100:+.1f} points "
              f"(95% CI {(res['est'] - half) * 100:+.1f} to {(res['est'] + half) * 100:+.1f}, "
              f"df {res['df']:.1f}, p = {res['p']:.3f}).\n")
        label = args.label or f"{args.world} data"
    else:
        vc = Components(parse_sds(args.variances), parse_sds(args.slope_variances), run=args.run_sd)
        label = args.label or "manual variances"
    changes = {k: v for k, v in (("baseline", args.baseline), ("checks_per_ticket", args.checks_per_ticket)) if v}
    if changes:
        vc = dataclasses.replace(vc, **changes)
    report(vc, sims=args.sims, seed=args.seed, label=label)


if __name__ == "__main__":
    main()
