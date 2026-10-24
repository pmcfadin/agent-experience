"""cpu_mem: Python batch jobs with one algorithmic CPU/memory regression (SPEC §4.5)."""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import random
from pathlib import Path
from string import Template

from elh.tasks.base import OracleBundle, TaskInstance, ValidationVariant

HERE = Path(__file__).parent

CAUSES = [
    "quadratic_membership",
    "nested_scan_join",
    "queue_pop_front",
    "repeated_sort_min",
    "exponential_recursion",
]

THEMES: dict[str, dict[str, str]] = {
    "inventory": {
        "prefix": "SKU", "key": "sku", "group": "warehouse", "value": "qty",
        "items": "stock_moves", "ref": "price_list", "verb": "reconcile_stock",
        "title": "Inventory reconciliation", "job": "nightly stock reconciliation",
    },
    "logistics": {
        "prefix": "PCL", "key": "tracking_id", "group": "depot", "value": "weight_g",
        "items": "parcels", "ref": "tariffs", "verb": "route_parcels",
        "title": "Parcel routing", "job": "parcel routing",
    },
    "telemetry": {
        "prefix": "SNS", "key": "sensor_id", "group": "site", "value": "reading_mv",
        "items": "readings", "ref": "calibrations", "verb": "aggregate_readings",
        "title": "Telemetry aggregation", "job": "telemetry aggregation",
    },
    "billing": {
        "prefix": "INV", "key": "invoice_id", "group": "account", "value": "amount_cents",
        "items": "invoices", "ref": "fx_rates", "verb": "settle_invoices",
        "title": "Invoice settlement", "job": "end-of-day settlement",
    },
    "course_registration": {
        "prefix": "STU", "key": "student_id", "group": "course", "value": "credits",
        "items": "enrollments", "ref": "fee_schedule", "verb": "process_enrollments",
        "title": "Enrollment processing", "job": "registration batch",
    },
    "network_flows": {
        "prefix": "FLW", "key": "flow_id", "group": "subnet", "value": "bytes_sent",
        "items": "flows", "ref": "qos_classes", "verb": "summarize_flows",
        "title": "Flow summarization", "job": "netflow summarizer",
    },
}

# Oracle parameters shared by every instance; part of the frozen task definition.
ORACLE_PARAMS = {
    "k_small": 3,
    "k_medium": 2,
    "correctness_timeout_s": 30.0,
    "public_test_timeout_s": 60.0,
    "reference_timeout_s": 120.0,
    "perf_factor": 10.0,
    "perf_floor_s": 3.0,
}

PROTECTED = ("run.py", "examples/sample_input.json", "examples/sample_output.json",
             "tests/test_job.py", "scripts/make_load.py")

UTIL_PY = '''\
"""Shared helpers for the ${title_lower} job."""


def is_valid(rec, field):
    value = rec.get(field)
    return isinstance(value, int) and value > 0


def id_number(identifier):
    return int(identifier.rsplit("-", 1)[1])
'''

RUN_PY = '''\
"""Entry point: python run.py INPUT.json OUTPUT.json"""

import json
import sys

from job.core import ${verb}


def main(argv):
    if len(argv) != 3:
        print("usage: python run.py INPUT.json OUTPUT.json", file=sys.stderr)
        return 2
    with open(argv[1]) as f:
        data = json.load(f)
    result = ${verb}(data)
    with open(argv[2], "w") as f:
        json.dump(result, f)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
'''

TEST_PY = '''\
import json
import os
import unittest

from job.core import ${verb}

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES = os.path.join(HERE, "..", "examples")


class SampleTest(unittest.TestCase):
    def test_sample_output_unchanged(self):
        with open(os.path.join(EXAMPLES, "sample_input.json")) as f:
            data = json.load(f)
        with open(os.path.join(EXAMPLES, "sample_output.json")) as f:
            expected = json.load(f)
        self.assertEqual(json.loads(json.dumps(${verb}(data))), expected)


if __name__ == "__main__":
    unittest.main()
'''

MAKE_LOAD_PY = '''\
"""Generate a synthetic, production-shaped input: python scripts/make_load.py N OUT.json [SEED]"""

import json
import random
import sys

F = ${fields_json}

${gen_source}

if __name__ == "__main__":
    n = int(sys.argv[1])
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    with open(sys.argv[2], "w") as f:
        json.dump(gen_input(random.Random(seed), n, F), f)
'''

README_MD = '''\
# ${title}

Batch job for the ${job} pipeline.

- Run: `python run.py INPUT.json OUTPUT.json`
- Tests: `python -m unittest discover -s tests`
- Synthetic load: `python scripts/make_load.py N OUT.json [SEED]`

Code lives in `job/`. `examples/` holds a small reference input and its expected output.
'''

TICKET = '''\
${opener}

${symptom}

Production runs see roughly ${large_n} ${items_human} per batch. You can produce a synthetic \
input of that shape with `python scripts/make_load.py N OUT.json`.

Requirements:
- Fix the root cause of the slowdown so the job handles production-sized input quickly.
- Output must stay byte-for-byte equivalent for every input (not just the sample), and the \
command-line interface (`python run.py INPUT.json OUTPUT.json`) must not change.
- Do not modify `run.py`, `examples/`, `tests/` or `scripts/`.

When you are done, call the `submit` tool with a short summary of the root cause and your fix.
'''

OPENERS = [
    "Ticket OPS-{n}: {title} job is too slow",
    "Incident follow-up #{n} — {title} performance regression",
    "Escalation {n}: {job} misses its batch window",
]

HARDCODE_CORE = '''\
import json
import os

_SAMPLE = os.path.join(os.path.dirname(__file__), "..", "examples", "sample_output.json")


def ${verb}(data):
    with open(_SAMPLE) as f:
        return json.load(f)
'''


def _cause_module(cause: str):
    return importlib.import_module(f"elh.tasks.families.cpu_mem.causes.{cause}")


def _render(template: str, F: dict[str, str]) -> str:
    return Template(template).substitute(F)


def _derive_seed(*parts: object) -> int:
    h = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(h[:8], "big")


class CpuMemFamily:
    name = "cpu_mem"
    causes = CAUSES
    themes = list(THEMES)

    def fields(self, theme: str) -> dict[str, str]:
        F = dict(THEMES[theme])
        F["title_lower"] = F["title"].lower()
        return F

    def _workspace(self, cause_mod, F: dict[str, str], core_template: str, seed: int) -> dict[str, str]:
        rng = random.Random(_derive_seed("sample", cause_mod.NAME, F["prefix"], seed))
        sample = cause_mod.gen_input(rng, cause_mod.SIZES["small"], F)
        expected = cause_mod.reference(sample, F)
        gen_source = inspect.getsource(cause_mod.gen_input)
        fields_json = json.dumps({k: F[k] for k in ("prefix", "key", "group", "value", "items", "ref")},
                                 indent=4)
        load_ctx = dict(F, fields_json=fields_json, gen_source=gen_source)
        return {
            "README.md": _render(README_MD, F),
            "run.py": _render(RUN_PY, F),
            "job/__init__.py": "",
            "job/core.py": _render(core_template, F),
            "job/util.py": _render(UTIL_PY, F),
            "tests/__init__.py": "",
            "tests/test_job.py": _render(TEST_PY, F),
            "scripts/make_load.py": Template(MAKE_LOAD_PY).substitute(load_ctx),
            "examples/sample_input.json": json.dumps(sample, indent=2, sort_keys=True) + "\n",
            "examples/sample_output.json": json.dumps(expected, indent=2, sort_keys=True) + "\n",
        }

    def generate(self, cause: str, theme: str, seed: int) -> TaskInstance:
        if cause not in CAUSES:
            raise KeyError(f"unknown cause {cause!r}")
        F = self.fields(theme)
        mod = _cause_module(cause)
        rng = random.Random(_derive_seed("surface", cause, theme, seed))
        task_id = f"{self.name}/{cause}/{theme}/{seed}"

        workspace = self._workspace(mod, F, mod.BUGGY_CORE, seed)
        opener = rng.choice(OPENERS).format(n=rng.randint(1000, 9999), **F)
        symptom = mod.SYMPTOM.format(items=F["items"].replace("_", " "), **{k: v for k, v in F.items() if k != "items"})
        prompt = Template(TICKET).substitute(
            opener=opener,
            symptom=symptom,
            large_n=f"{mod.SIZES['large']:,}",
            items_human=F["items"].replace("_", " "),
        )

        params = dict(ORACLE_PARAMS, hidden_seed=_derive_seed("oracle", task_id))
        oracle = OracleBundle(
            files={
                "oracle.py": (HERE / "oracle_runner.py").read_text(),
                "cause.py": Path(inspect.getsourcefile(mod)).read_text(),
                "fields.json": json.dumps(F, sort_keys=True),
                "params.json": json.dumps(params, sort_keys=True),
            },
            command=["python3", "{oracle}/oracle.py", "--workspace", "{workspace}", "--oracle", "{oracle}"],
        )
        return TaskInstance(
            task_id=task_id, family=self.name, cause=cause, theme=theme, seed=seed, difficulty=1,
            prompt=prompt, workspace=workspace, protected=PROTECTED, oracle=oracle,
            signature={"family": self.name, "cause": cause, "theme": theme,
                       "symptom": "cpu_bound_slowdown", "language": "python"},
        )

    def solution_core(self, task: TaskInstance) -> str:
        """Correct fix for job/core.py (used by validation and the scripted 'reference' agent)."""
        return _render(_cause_module(task.cause).FIXED_CORE, self.fields(task.theme))

    def wrong_fix_core(self, task: TaskInstance) -> str:
        return _render(_cause_module(task.cause).WRONG_FIX_CORE, self.fields(task.theme))

    def validation_variants(self, task: TaskInstance) -> list[ValidationVariant]:
        F = self.fields(task.theme)

        def with_core(src: str) -> dict[str, str]:
            return dict(task.workspace, **{"job/core.py": src})

        return [
            ValidationVariant("reference_fix", with_core(self.solution_core(task)), True),
            ValidationVariant("untouched_buggy", dict(task.workspace), False, "performance"),
            ValidationVariant("wrong_fix", with_core(self.wrong_fix_core(task)), False, "correctness"),
            ValidationVariant("hardcoded_sample", with_core(_render(HARDCODE_CORE, F)), False, "correctness"),
        ]
