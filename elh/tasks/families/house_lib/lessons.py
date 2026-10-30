"""Lesson templates and world catalogues (design §3).

A lesson is a template instantiated for one context (a partner, region or module) with an arbitrary
choice drawn from the world seed. Each template knows how to state the lesson (gold text, `source`
table entry, `doc` sentence), how a solution applies it (code lines) and how a solution that ignores
it looks (the blind variant). The oracle checks lessons from the same `template` + `choice` data.

Implemented: 7 of the design's 10 templates. `counterfactual_default`, `sentinel_return` and
`retry_policy` are not (they need paging, cache and fault injection in the feature contract).

Conflicts (Decision): with conflict rate c, round(c·L/2) pairs of same-template lessons get two
different choices (a `conflict_group`); every other lesson of a template shares one template-wide
choice. So at c = 0 a template's lessons agree and a stated meta-lesson ("partner feeds use
dmy_dot") covers them; at c > 0 rules of the same template contradict across contexts.
"""

from __future__ import annotations

import random
import re
from dataclasses import asdict, dataclass

PARTNERS = ["halden", "orsk", "varna", "tellis", "mirro", "quent", "sabel", "dorrin",
            "kessa", "lumo", "brant", "ystad", "corvo", "elm", "fjord", "nadir"]
REGIONS = ["eu", "us", "uk", "apac", "latam", "ca", "mena", "nordics"]
MODULES = ["orders", "ledger", "catalog", "payments", "shipping", "payroll", "crm", "reporting"]
CONTEXTS = {"partner": PARTNERS, "region": REGIONS, "module": MODULES}

IDEM_PARTS = {"key": ("key",), "group+key": ("group", "key"), "module+key": ("module", "key"),
              "key+due": ("key", "due")}


@dataclass(frozen=True)
class Template:
    name: str
    context: str              # partner | region | module
    field: str | None         # output column the lesson governs; None = module-wide behaviour
    choices: tuple[str, ...]
    table: str                # placeholder of the `source` table in acmecore (library.py)
    topic: str                # acmecore/docs/<topic> that carries `doc` lessons


TEMPLATES = {t.name: t for t in [
    Template("partner_format", "partner", "due_date", ("dmy_dot", "ymd_slash", "epoch_ms", "mdy_dash", "yyyymmdd"),
             "partner_styles", "partners.md"),
    Template("unit_convention", "module", "amount", ("cents", "decimal_str", "milli", "cents_str"),
             "module_units", "money.md"),
    Template("region_redaction", "region", "contact", ("mask", "digest", "domain"), "region_contact", "privacy.md"),
    Template("required_wrapper", "module", "generated_at", ("batch_time", "cutoff", "settled_at"),
             "module_clocks", "time.md"),
    Template("event_fields", "module", "idem_key", tuple(IDEM_PARTS), "module_idem_parts", "events.md"),
    Template("call_order", "module", None, ("before", "after"), "module_audit", "audit.md"),
    Template("feature_flag", "module", None, ("partner_feeds_v2", "exports_2026", "new_writer", "feeds_rollout"),
             "module_flags", "rollout.md"),
]}

# Templates without an output column apply to every task in their module (design §4.2 needs them
# whenever the module writes, which every feature does).
MODULE_WIDE = tuple(n for n, t in TEMPLATES.items() if t.field is None)


@dataclass(frozen=True)
class Lesson:
    id: str                   # <template>.<context value>
    template: str
    context: str
    choice: str
    visibility: str           # source | doc | hidden
    conflict_group: str | None

    @property
    def tpl(self) -> Template:
        return TEMPLATES[self.template]

    def to_dict(self) -> dict:
        return asdict(self)


def gold_text(lesson: Lesson) -> str:
    """One-sentence statement of the lesson (handbook and oracle controls; also the `doc` prose)."""
    c, x, t = lesson.choice, lesson.context, lesson.template
    if t == "partner_format":
        return f"Feeds for partner {x} format dates with `formats.format_date(d, \"{c}\")`."
    if t == "unit_convention":
        return f"In the `{x}` module, amounts are exported as `units.convert(cents, \"{c}\")`."
    if t == "region_redaction":
        return f"Contact details of {x.upper()} customers are exported as `pii.{c}(email)`."
    if t == "required_wrapper":
        return f"Timestamps written by the `{x}` module come from `clock.{c}()`."
    if t == "event_fields":
        return (f"Rows written by the `{x}` module carry `idem_key = ids.idem({', '.join(IDEM_PARTS[c])})`, "
                f"parts in that order (`module` is the module name).")
    if t == "call_order":
        return f"In the `{x}` module, `audit.record(table, key)` is called immediately {c} each `store.write`."
    if t == "feature_flag":
        return f"New code paths in the `{x}` module are gated with `flags.on(\"{c}\", \"{x}\")` and write nothing when it is off."
    raise KeyError(t)


def source_entry(lesson: Lesson):
    """(key, value) of the lesson's row in its acmecore lookup table."""
    if lesson.template == "event_fields":
        return lesson.context, IDEM_PARTS[lesson.choice]
    return lesson.context, lesson.choice


@dataclass(frozen=True)
class World:
    id: str                   # L<lessons>-c<conflict %>-w<seed>[-h]
    n_lessons: int
    conflict_rate: float
    seed: int
    with_hidden: bool
    lessons: tuple[Lesson, ...]

    def lesson(self, lesson_id: str) -> Lesson:
        for les in self.lessons:
            if les.id == lesson_id:
                return les
        raise KeyError(f"lesson {lesson_id!r} is not in world {self.id}")

    def sibling(self, lesson: Lesson) -> Lesson | None:
        if lesson.conflict_group is None:
            return None
        return next(x for x in self.lessons if x.conflict_group == lesson.conflict_group and x.id != lesson.id)


WORLD_ID = re.compile(r"^L(\d+)-c(\d+)-w(\d+)(-h)?$")


def parse_world_id(world_id: str) -> tuple[int, float, int, bool]:
    m = WORLD_ID.match(world_id)
    if not m:
        raise ValueError(f"world id {world_id!r} must look like L16-c50-w7 (optional -h: include hidden lessons)")
    return int(m[1]), int(m[2]) / 100, int(m[3]), bool(m[4])


def build_world(world_id: str) -> World:
    """The world's lesson catalogue, deterministic from its id."""
    n, c, seed, with_hidden = parse_world_id(world_id)
    pool = [(t, x) for t, tpl in TEMPLATES.items() for x in CONTEXTS[tpl.context]]
    if not 1 <= n <= len(pool):
        raise ValueError(f"lessons per world must be in [1, {len(pool)}], got {n}")
    rng = random.Random(f"house_lib/world/{seed}/{n}/{c}")
    base = {t: rng.choice(tpl.choices) for t, tpl in TEMPLATES.items()}

    # Conflict pairs first (two contexts of one template, two different choices), then fill.
    chosen: dict[tuple[str, str], tuple[str, str | None]] = {}
    for g in range(round(c * n / 2)):
        open_t = [t for t in TEMPLATES if sum((t, x) not in chosen for x in CONTEXTS[TEMPLATES[t].context]) >= 2]
        t = rng.choice(open_t)
        x1, x2 = rng.sample([x for x in CONTEXTS[TEMPLATES[t].context] if (t, x) not in chosen], 2)
        a, b = rng.sample(TEMPLATES[t].choices, 2)
        chosen[t, x1], chosen[t, x2] = (a, f"g{g}"), (b, f"g{g}")
    rest = [p for p in pool if p not in chosen]
    for t, x in rng.sample(rest, n - len(chosen)):
        chosen[t, x] = (base[t], None)

    keys = sorted(chosen)
    levels = ["source", "doc", "hidden"] if with_hidden else ["source", "doc"]
    vis = [levels[i % len(levels)] for i in range(len(keys))]
    rng.shuffle(vis)
    lessons = tuple(Lesson(f"{t}.{x}", t, x, chosen[t, x][0], v, chosen[t, x][1])
                    for (t, x), v in zip(keys, vis))
    return World(world_id, n, c, seed, with_hidden, lessons)
