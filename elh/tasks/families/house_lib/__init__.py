"""house_lib: feature tickets against a generated in-house library with house conventions
(docs/design/house-lib-family.md).

One family object per world: `get_family("house_lib/L16-c50-w7")`. Its `causes` are the world's
lesson ids; a task's cause is its primary lesson and `signature["lessons"]` lists every lesson it
requires. TaskInstance.family carries the world (`house_lib/<world>`) so a stored episode
regenerates through `get_family(row["family"])` like cpu_mem's.

A task asks for one export: `export(records)` in `service/<module>/<partner>_<items>_export.py`
writes one row per qualifying record (region and minimum amount) with `acmecore.store.write`.
Lesson-neutral columns are the record's key and group; each required lesson adds a column or a
module-wide behaviour (audit order, feature flag) that the ticket never spells out.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from string import Template

from elh.tasks.base import OracleBundle, TaskInstance, ValidationVariant
from elh.tasks.families.cpu_mem import THEMES
from elh.tasks.families.house_lib import library
from elh.tasks.families.house_lib.lessons import (IDEM_PARTS, MODULE_WIDE, MODULES, PARTNERS, REGIONS, TEMPLATES,
                                                  Lesson, World, build_world, gold_text, source_entry)
from elh.tasks.families.house_lib.records import SIZES, gen_records

HERE = Path(__file__).parent
NAME = "house_lib"
DEFAULT_WORLD = "L16-c50-w1"

# Lessons per task m: 1-3, mean 1.5 (design §7.1). Module-wide lessons of the task's module come on
# top when the module has them; the generator prefers modules without any.
M_WEIGHTS = {1: 0.6, 2: 0.3, 3: 0.1}

MIN_CENTS = [500, 1000, 2500, 5000]

ORACLE_PARAMS = {
    "k_small": 3,
    "k_medium": 2,
    "case_timeout_s": 30.0,
    "public_test_timeout_s": 60.0,
    "frozen_clock": {"now": "2026-03-14T09:26:53+00:00", "batch_time": "2026-03-14T02:00:00+00:00",
                     "cutoff": "2026-03-14T17:00:00+00:00", "settled_at": "2026-03-14T09:00:00+00:00"},
}

COLUMN_TEXT = {
    "due_date": "`due_date`: the record's due date",
    "amount": "`amount`: the outstanding amount",
    "contact": "`contact`: the customer's email contact",
    "generated_at": "`generated_at`: when the export ran",
    "idem_key": "`idem_key`: an idempotency key for the row",
}

OPENERS = [
    "Ticket FEAT-{n}: {Partner} {items_human} feed",
    "Request #{n} from partner success: {Partner} needs a {items_human} export",
    "FEAT-{n}: new partner export for {Partner}",
]

TICKET = '''\
${opener}

Partner ${Partner} needs a nightly export of ${items_human} for ${REGION} accounts with at least \
${min_cents} cents outstanding (`amount_cents >= ${min_cents}`, `region == "${region}"`).

Implement `export(records)` in `${feature_path}`. For each qualifying record, in input order, write \
one row to the `${table}` table with `acmecore.store.write`. Each row has these columns:
${columns}

`examples/` has a sample input and the expected rows (key columns only). The `${module}` module is \
part of the Acme service codebase: follow its conventions.

Requirements:
- Only change files under `service/`.
- Do not modify `acmecore/`, `examples/` or `tests/`.

When you are done, call the `submit` tool with a short summary of what you implemented.
'''

README_MD = '''\
# Acme services

Service code lives in `service/<module>/`; shared building blocks in `acmecore/` (docs in
`acmecore/docs/`). Partner and region registry: `service/partners.py`.

- Tests: `python -m unittest discover -s tests`
'''

PARTNERS_PY = '''\
"""Partner and region registry."""

PARTNERS = ${partners}
REGIONS = ${regions}
MODULES = ${modules}
'''

TEST_PY = '''\
import json
import os
import unittest

from acmecore import store
from ${feature_mod} import export

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES = os.path.join(HERE, "..", "examples")
KEYS = (${key_repr}, ${group_repr})


class SampleTest(unittest.TestCase):
    def test_sample_rows(self):
        with open(os.path.join(EXAMPLES, "sample_input.json")) as f:
            records = json.load(f)
        with open(os.path.join(EXAMPLES, "sample_output.json")) as f:
            expected = json.load(f)
        store.reset()
        export(records)
        got = [{k: row[k] for k in KEYS} for row in store.rows(${table_repr})]
        self.assertEqual(got, expected)


if __name__ == "__main__":
    unittest.main()
'''

STUB_PY = '''\
"""${Partner} ${items_human} export (${module} module)."""

from acmecore import store

TABLE = "${table}"


def export(records):
    raise NotImplementedError("see TICKET.md")
'''

HARDCODE_PY = '''\
import json
import os

from acmecore import store

TABLE = "${table}"
_SAMPLE = os.path.join(os.path.dirname(__file__), "..", "..", "examples", "sample_output.json")


def export(records):
    with open(_SAMPLE) as f:
        for row in json.load(f):
            store.write(TABLE, row)
'''


def _derive_seed(*parts: object) -> int:
    h = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(h[:8], "big")


def _idem_expr(choice: str, spec: dict) -> str:
    src = {"key": f'rec["{spec["key"]}"]', "group": f'rec["{spec["group"]}"]', "module": repr(spec["module"]),
           "due": 'rec["due"]'}
    return ", ".join(src[p] for p in IDEM_PARTS[choice])


def render_feature(spec: dict, lessons: dict[str, Lesson], *, blind: str | None = None,
                   swap: dict[str, str] | None = None) -> str:
    """The feature module that applies every required lesson. `blind` = a lesson id to ignore (the
    necessity variant); `swap` = {lesson id: choice} to apply another choice (the conflict variant)."""
    swap = swap or {}
    key, group, module = spec["key"], spec["group"], spec["module"]

    def choice(les: Lesson) -> str:
        return swap.get(les.id, les.choice)

    by_t = {les.template: les for les in lessons.values()}
    on = {t: les for t, les in by_t.items() if les.id != blind}
    lines = []
    if "partner_format" in by_t:
        lines.append(f'row["due_date"] = formats.format_date(rec["due"], "{choice(on["partner_format"])}")'
                     if "partner_format" in on else 'row["due_date"] = rec["due"]')
    if "unit_convention" in by_t:
        lines.append(f'row["amount"] = units.convert(rec["amount_cents"], "{choice(on["unit_convention"])}")'
                     if "unit_convention" in on else 'row["amount"] = rec["amount_cents"] / 100')
    if "region_redaction" in by_t:
        lines.append(f'row["contact"] = pii.{choice(on["region_redaction"])}(rec["email"])'
                     if "region_redaction" in on else 'row["contact"] = rec["email"]')
    if "required_wrapper" in by_t:
        lines.append(f'row["generated_at"] = clock.{choice(on["required_wrapper"])}()'
                     if "required_wrapper" in on else 'row["generated_at"] = clock.now()')
    if "event_fields" in on:  # blind: no key at all
        lines.append(f'row["idem_key"] = ids.idem({_idem_expr(choice(on["event_fields"]), spec)})')
    audit = 'audit.record(TABLE, row["' + key + '"])'
    order = choice(on["call_order"]) if "call_order" in on else None
    if order == "before":
        lines.append(audit)
    lines.append("store.write(TABLE, row)")
    if order == "after":
        lines.append(audit)
    gate = ""
    if "feature_flag" in on:
        gate = f'    if not flags.on("{choice(on["feature_flag"])}", "{module}"):\n        return\n'
    body = "".join(f"        {s}\n" for s in lines)
    return (f'"""{spec["Partner"]} {spec["items_human"]} export ({module} module)."""\n\n'
            "from acmecore import audit, clock, flags, formats, ids, pii, store, units\n\n"
            f'TABLE = "{spec["table"]}"\n\n\n'
            "def export(records):\n"
            f"{gate}"
            "    for rec in records:\n"
            f'        if rec["region"] != "{spec["region"]}" or rec["amount_cents"] < {spec["min_cents"]}:\n'
            "            continue\n"
            f'        row = {{"{key}": rec["{key}"], "{group}": rec["{group}"]}}\n'
            f"{body}")


class HouseLibFamily:
    themes = list(THEMES)

    def __init__(self, world_id: str = DEFAULT_WORLD):
        self.world: World = build_world(world_id)
        self.name = f"{NAME}/{world_id}"
        self.causes = [les.id for les in self.world.lessons]
        self._library = self._build_library()
        self.library_sha256 = library.fingerprint(self._library)

    def _build_library(self) -> dict[str, str]:
        tables: dict[str, dict] = {t.table: {} for t in TEMPLATES.values()}
        docs: dict[str, list[str]] = {}
        for les in self.world.lessons:
            if les.visibility == "source":
                k, v = source_entry(les)
                tables[les.tpl.table][k] = v
            elif les.visibility == "doc":
                docs.setdefault(les.tpl.topic, []).append(gold_text(les))
        return library.build_library(tables, docs)

    # -- task selection -------------------------------------------------------------------------

    def _select(self, primary: Lesson, rng: random.Random) -> tuple[dict, list[Lesson]]:
        """Context (module, partner, region) and the required lessons for a task."""
        ctx = {primary.tpl.context: primary.context}
        chosen = [primary]
        m = rng.choices(list(M_WEIGHTS), weights=list(M_WEIGHTS.values()))[0]
        cands = [les for les in self.world.lessons if les.template not in MODULE_WIDE]
        rng.shuffle(cands)
        for les in cands:
            if len(chosen) >= m:
                break
            if les.template in {x.template for x in chosen}:
                continue
            if ctx.get(les.tpl.context, les.context) != les.context:
                continue
            chosen.append(les)
            ctx[les.tpl.context] = les.context
        if "module" not in ctx:
            wide = {x: sum(1 for les in self.world.lessons if les.template in MODULE_WIDE and les.context == x)
                    for x in MODULES}
            quiet = [x for x in MODULES if wide[x] == 0]
            ctx["module"] = rng.choice(quiet or MODULES)
        ctx.setdefault("partner", rng.choice(PARTNERS))
        ctx.setdefault("region", rng.choice(REGIONS))
        for les in self.world.lessons:  # module-wide lessons of the task's module
            if les.template in MODULE_WIDE and les.context == ctx["module"] and les not in chosen:
                chosen.append(les)
        return ctx, chosen

    def spec(self, task: TaskInstance) -> dict:
        return json.loads(task.oracle.files["params.json"])["spec"]

    def required(self, task: TaskInstance) -> dict[str, Lesson]:
        return {lid: self.world.lesson(lid) for lid in task.signature["lessons"]}

    # -- generation -----------------------------------------------------------------------------

    def generate(self, cause: str, theme: str, seed: int) -> TaskInstance:
        primary = self.world.lesson(cause)
        F = THEMES[theme]
        task_id = f"{self.name}/{cause}/{theme}/{seed}"
        rng = random.Random(_derive_seed("house_lib", task_id))
        ctx, lessons = self._select(primary, rng)
        items_human = F["items"].replace("_", " ")
        table = f'{ctx["partner"]}_{F["items"]}'
        rel = f'service/{ctx["module"]}/{ctx["partner"]}_{F["items"]}_export.py'
        spec = {"key": F["key"], "group": F["group"], "prefix": F["prefix"], "items_human": items_human,
                "module": ctx["module"], "partner": ctx["partner"], "Partner": ctx["partner"].capitalize(),
                "region": ctx["region"], "REGION": ctx["region"].upper(), "min_cents": rng.choice(MIN_CENTS),
                "table": table, "feature_path": rel, "feature_mod": rel[:-3].replace("/", ".")}
        columns = [f"- `{F['key']}`: the record's {F['key']}", f"- `{F['group']}`: the record's {F['group']}"]
        columns += [f"- {COLUMN_TEXT[les.tpl.field]}" for les in sorted(lessons, key=lambda x: x.template)
                    if les.tpl.field]
        opener = rng.choice(OPENERS).format(n=rng.randint(1000, 9999), **spec)
        prompt = Template(TICKET).substitute(spec, opener=opener, columns="\n".join(columns))

        sample = gen_records(random.Random(_derive_seed("sample", task_id)), SIZES["sample"], F,
                             spec["region"], spec["min_cents"])
        expected = [{F["key"]: r[F["key"]], F["group"]: r[F["group"]]} for r in sample
                    if r["region"] == spec["region"] and r["amount_cents"] >= spec["min_cents"]]
        workspace = dict(self._library)
        workspace.update({
            "README.md": README_MD,
            "TICKET.md": prompt,
            "service/__init__.py": "",
            "service/partners.py": Template(PARTNERS_PY).substitute(
                partners=repr(PARTNERS), regions=repr(REGIONS), modules=repr(MODULES)),
            f'service/{ctx["module"]}/__init__.py': "",
            rel: Template(STUB_PY).substitute(spec),
            "tests/__init__.py": "",
            "tests/test_public.py": Template(TEST_PY).substitute(
                spec, key_repr=repr(F["key"]), group_repr=repr(F["group"]), table_repr=repr(table)),
            "examples/sample_input.json": json.dumps(sample, indent=2) + "\n",
            "examples/sample_output.json": json.dumps(expected, indent=2) + "\n",
        })
        protected = tuple(sorted(p for p in workspace if p.startswith(("acmecore/", "examples/")))) \
            + ("tests/test_public.py",)

        params = dict(ORACLE_PARAMS, hidden_seed=_derive_seed("oracle", task_id), spec=spec,
                      sizes=SIZES, lessons=[les.to_dict() for les in lessons])
        oracle_files = {f"acmecore/{p.removeprefix('acmecore/')}": src for p, src in self._library.items()}
        oracle_files.update({
            "oracle.py": (HERE / "oracle_runner.py").read_text(),
            "records.py": (HERE / "records.py").read_text(),
            "fields.json": json.dumps(F, sort_keys=True),
            "params.json": json.dumps(params, sort_keys=True),
        })
        oracle = OracleBundle(
            files=oracle_files,
            command=["python3", "{oracle}/oracle.py", "--workspace", "{workspace}", "--oracle", "{oracle}"],
        )
        return TaskInstance(
            task_id=task_id, family=self.name, cause=cause, theme=theme, seed=seed, difficulty=len(lessons),
            prompt=prompt, workspace=workspace, protected=protected, oracle=oracle,
            signature={"family": NAME, "world": self.world.id, "library_sha256": self.library_sha256,
                       "cause": cause, "theme": theme, "lessons": [les.id for les in lessons],
                       "templates": sorted({les.template for les in lessons}),
                       "visibility": {les.id: les.visibility for les in lessons},
                       "conflict_groups": {les.id: les.conflict_group for les in lessons if les.conflict_group},
                       "context": ctx, "language": "python"},
        )

    # -- solutions and validation variants ------------------------------------------------------

    def solution_files(self, task: TaskInstance) -> dict[str, str]:
        """Reference solution (validation and the scripted 'reference' agent)."""
        spec = self.spec(task)
        return {spec["feature_path"]: render_feature(spec, self.required(task))}

    def wrong_files(self, task: TaskInstance) -> dict[str, str]:
        """The scripted 'wrong' agent: does the feature but ignores the primary lesson."""
        spec = self.spec(task)
        return {spec["feature_path"]: render_feature(spec, self.required(task), blind=task.cause)}

    def validation_variants(self, task: TaskInstance) -> list[ValidationVariant]:
        spec = self.spec(task)
        lessons = self.required(task)

        def with_feature(src: str) -> dict[str, str]:
            return dict(task.workspace, **{spec["feature_path"]: src})

        out = [
            ValidationVariant("reference_fix", with_feature(render_feature(spec, lessons)), True),
            ValidationVariant("untouched_stub", dict(task.workspace), False, "feature"),
            ValidationVariant("hardcoded_sample", with_feature(Template(HARDCODE_PY).substitute(spec)), False, "feature"),
        ]
        for lid, les in lessons.items():
            out.append(ValidationVariant(f"blind:{les.template}", with_feature(render_feature(spec, lessons, blind=lid)),
                                         False, f"lesson:{lid}", exact=True))
            sib = self.world.sibling(les)
            if sib is not None:
                out.append(ValidationVariant(f"conflict:{les.template}",
                                             with_feature(render_feature(spec, lessons, swap={lid: sib.choice})),
                                             False, f"lesson:{lid}", exact=True))
        return out
