"""house_lib task family (docs/design/house-lib-family.md): worlds, tasks, oracle, validation."""

import json
from collections import Counter
from pathlib import Path

import pytest
import yaml

from elh.batch import load_experiment, run_batch, schedule
from elh.controller import regrade_episode, run_single_episode
from elh.judge import grade
from elh.registry import load_config
from elh.runner.scripts import scripted_client
from elh.sandbox.local import LocalSandbox
from elh.store import Store
from elh.tasks.base import get_family
from elh.tasks.families.house_lib import calibration
from elh.tasks.families.house_lib.lessons import MODULE_WIDE, build_world, gold_text
from elh.validation import validate_family

CONFIG = Path(__file__).parent.parent / "experiments" / "phase0-local.yaml"
WORLD = "house_lib/L16-c50-w1"
TABLE_FILE = {"partner_format": "formats.py", "unit_convention": "units.py", "region_redaction": "pii.py",
              "required_wrapper": "clock.py", "event_fields": "ids.py", "call_order": "audit.py",
              "feature_flag": "flags.py"}


@pytest.fixture(scope="module")
def family():
    return get_family(WORLD)


def test_world_is_deterministic_with_the_requested_shape():
    w = build_world("L16-c50-w1")
    assert w == build_world("L16-c50-w1") and w != build_world("L16-c50-w2")
    assert len(w.lessons) == 16 and len({les.id for les in w.lessons}) == 16
    groups = Counter(les.conflict_group for les in w.lessons if les.conflict_group)
    assert sum(groups.values()) == 8 and set(groups.values()) == {2}  # c = 0.5 -> 4 pairs
    for g in groups:
        a, b = [les for les in w.lessons if les.conflict_group == g]
        assert a.template == b.template and a.context != b.context and a.choice != b.choice
    assert {les.visibility for les in w.lessons} == {"source", "doc"}  # main grid: no hidden lessons
    assert "hidden" in {les.visibility for les in build_world("L16-c0-w1-h").lessons}
    # c = 0: a template's lessons share one choice (no contradictions between rules).
    flat = build_world("L64-c0-w5")
    assert all(len({les.choice for les in flat.lessons if les.template == t}) == 1
               for t in {les.template for les in flat.lessons})
    with pytest.raises(ValueError):
        build_world("L65-c0-w1")


def test_library_states_source_and_doc_lessons_only_where_designed(family):
    task = family.generate(family.causes[0], "billing", 1)
    lib = {p: s for p, s in task.workspace.items() if p.startswith("acmecore/")}
    docs = "".join(s for p, s in lib.items() if p.startswith("acmecore/docs/"))
    for les in family.world.lessons:
        table_file = lib[f"acmecore/{TABLE_FILE[les.template]}"]
        assert (f"    {les.context!r}:" in table_file) == (les.visibility == "source"), les.id
        assert (gold_text(les) in docs) == (les.visibility == "doc"), les.id
    # Identical library in every task of the world; it is in the signature.
    other = family.generate(family.causes[-1], "telemetry", 9)
    assert {p: s for p, s in other.workspace.items() if p.startswith("acmecore/")} == lib
    assert task.signature["library_sha256"] == other.signature["library_sha256"] == family.library_sha256


def test_tasks_are_deterministic_and_never_name_the_convention(family):
    for cause in family.causes:
        a, b = family.generate(cause, "logistics", 3), family.generate(cause, "logistics", 3)
        assert a.content_hash == b.content_hash and a.family == WORLD
        assert a.task_id == f"{WORLD}/{cause}/logistics/3" and a.signature["lessons"][0] == cause
        lessons = [family.world.lesson(lid) for lid in a.signature["lessons"]]
        assert len({les.template for les in lessons}) == len(lessons)
        for les in lessons:  # choices that are identifiers; "cents"/"key"/"after" are ordinary words
            if les.template in ("partner_format", "region_redaction", "required_wrapper", "feature_flag"):
                assert les.choice not in a.prompt, (cause, les.id)
            assert gold_text(les) not in a.prompt
        # Every module-wide lesson of the task's module is required.
        module = a.signature["context"]["module"]
        wide = {les.id for les in family.world.lessons if les.template in MODULE_WIDE and les.context == module}
        assert wide <= set(a.signature["lessons"])
    assert family.generate(family.causes[0], "logistics", 4).content_hash != \
        family.generate(family.causes[0], "logistics", 3).content_hash


def test_oracle_and_hidden_inputs_never_in_workspace(family):
    task = family.generate(family.causes[3], "inventory", 2)
    assert "oracle.py" in task.oracle.files and "params.json" in task.oracle.files
    assert not any(p.startswith("oracle") or p == "records.py" for p in task.workspace)
    assert "acmecore/store.py" in task.protected and "tests/test_public.py" in task.protected


def test_validation_passes_for_a_slice_of_the_world(family):
    results = validate_family(family, lambda: LocalSandbox(), seeds=[1], themes=["billing"], workers=6)
    bad = [r for r in results if not r.ok]
    assert not bad, [(r.task_id, r.variant, r.reason) for r in bad[:5]]
    kinds = Counter(r.variant.split(":")[0] for r in results)
    assert kinds["blind"] >= len(family.causes) and kinds["conflict"] > 0


def test_necessity_blind_variant_fails_exactly_its_lesson(family):
    lid = next(les.id for les in family.world.lessons if les.template == "partner_format")
    task = family.generate(lid, "network_flows", 7)
    variants = {v.name: v for v in family.validation_variants(task)}
    verdict = grade(task, variants["blind:partner_format"].workspace, lambda: LocalSandbox())
    assert verdict.failed_checks() == [f"lesson:{lid}"]
    assert verdict.checks["feature"] and verdict.checks["public_tests"]


@pytest.mark.parametrize("template", ["partner_format", "call_order", "feature_flag"])
def test_feedback_names_what_failed_but_not_the_lesson(family, template):
    """Memory records, their reranked context and the rule writer's check block describe a failed
    lesson check in plain words: never the lesson id, its template, its choice or the world id."""
    from types import SimpleNamespace

    from elh.learner.rules import _checks_block
    from elh.memory.hindsight import lesson_v3_item, render_context

    les = next(les for les in family.world.lessons if les.template == template)
    task = family.generate(les.id, "network_flows", 7)
    variants = {v.name: v for v in family.validation_variants(task)}
    verdict = grade(task, variants[f"blind:{template}"].workspace, lambda: LocalSandbox())
    assert verdict.failed_checks() == [f"lesson:{les.id}"]
    ep = {"family": task.family, "prompt": task.prompt, "verdict": verdict.to_dict(), "outcome": "submitted",
          "turns": 5, "tool_calls": 6, "submit_summary": "done", "diff": None, "episode_id": "e1",
          "rule": {"kind": "avoid", "when": "w", "then": "t"}}
    item = lesson_v3_item(ep, ticket_metadata=False)
    texts = [item["content"], item["context"], render_context(ep),
             _checks_block(verdict, SimpleNamespace(outcome="submitted", turns=5, tool_calls=6))]
    # call_order's choices are plain words ("after" also appears in "submitted its fix after 5 turns").
    choice = () if template == "call_order" else (les.choice,)
    for text in texts:
        for leak in ("lesson:", les.id, template, family.world.id, "performance", "fast enough") + choice:
            assert leak not in text
    phrase = verdict.detail["check_text"][f"lesson:{les.id}"]
    assert f"FAILED: {phrase['failed']}." in item["content"]
    assert item["context"].startswith(f"Verified result of a coding agent's attempt at a house_lib ticket: "
                                      f"FAILED ({phrase['label']})")


def test_editing_acmecore_fails_integrity_and_is_not_used_by_the_oracle(family):
    task = family.generate(family.causes[0], "billing", 1)
    ws = dict(task.workspace, **family.solution_files(task))
    ws["acmecore/clock.py"] = ws["acmecore/clock.py"] + "\n# changed\n"
    verdict = grade(task, ws, lambda: LocalSandbox())
    assert verdict.failed_checks() == ["integrity"]


@pytest.mark.parametrize("kind,expect_pass", [("reference", True), ("wrong", False), ("noop", False)])
def test_scripted_episode_end_to_end_and_replay(tmp_path, family, kind, expect_pass):
    cfg = load_config(CONFIG)
    store = Store(tmp_path)
    task = family.generate(family.causes[5], "course_registration", 4)
    client = scripted_client(kind, task, family)
    episode_id, result, verdict = run_single_episode(cfg, store, task, client, sandbox_backend="local")
    assert result.outcome == "submitted" and verdict.passed is expect_pass
    if kind == "wrong":
        assert verdict.failed_checks() == [f"lesson:{task.cause}"]
    row = store.episode(episode_id)
    assert row["family"] == WORLD and row["cause"] == task.cause
    same, _, _ = regrade_episode(cfg, store, episode_id, sandbox_backend="local")
    assert same


def test_calibration_panel_and_filters(family):
    tasks = calibration.calibration_tasks(family, per_lesson=2)
    assert len(tasks) == 2 * len(family.causes) and all(3000 <= s < 4000 for _, _, s in tasks)
    assert len({s for _, _, s in tasks}) == len(tasks)
    text = calibration.handbook(family)
    assert text.count("\n") == len(family.causes) - 1
    eps = ([{"group": "A", "checks": {"lesson:x": True}}] * 3 + [{"group": "A", "checks": {"lesson:x": False}}] * 7
           + [{"group": "P-hand", "checks": {"lesson:x": True, "lesson:y": False}}] * 2
           + [{"group": "A", "checks": {"lesson:y": False}}])
    res = {r.lesson: r for r in calibration.filter_lessons(eps, ["x", "y", "z"])}
    assert not res["x"].keep and "guessable" in res["x"].reason          # A passes 30%
    assert not res["y"].keep and "not learnable" in res["y"].reason      # P-hand passes 0%
    assert res["z"].keep and "undecided" in res["z"].reason


def test_batch_runs_a_house_lib_world(tmp_path):
    lessons = [les.id for les in build_world("L4-c0-w2").lessons]
    doc = {"name": "hl-batch", "base_config": str(CONFIG), "family": "house_lib/L4-c0-w2", "histories": 1,
           "groups": ["A"], "phases": [{"eval": {"tasks": {"causes": lessons[:2], "themes": ["billing"],
                                                           "seeds": [2000, 2001]}}}]}
    path = tmp_path / "hl-batch.yaml"
    path.write_text(yaml.safe_dump(doc))
    exp, cfg = load_experiment(path)
    fam = get_family(exp.family)
    results = run_batch(exp, cfg, Store(tmp_path / "runs"), lambda t: scripted_client("reference", t, fam),
                        sandbox_backend="local", out=lambda _: None)
    assert [r["passed"] for r in results] == [True, True]
    bad = tmp_path / "bad.yaml"
    doc["phases"][0]["eval"]["tasks"]["causes"] = ["partner_format.nowhere"]  # not a lesson of this world
    bad.write_text(yaml.safe_dump(doc))
    with pytest.raises(ValueError, match="unknown causes"):
        load_experiment(bad)


def test_p_hand_shows_the_world_handbook_and_never_trains(tmp_path):
    from pydantic import ValidationError

    from elh.runner.prompts import system_prompt, tools

    world = "house_lib/L4-c0-w2"
    lessons = [les.id for les in build_world("L4-c0-w2").lessons]
    doc = {"name": "hl-hand", "base_config": str(CONFIG), "family": world, "histories": 1,
           "groups": ["P-hand", "A0", "B"], "overrides": {"memory_service.agent_search": True},
           "phases": [{"eval": {"tasks": {"causes": lessons[:1], "themes": ["billing"], "seeds": [2000, 2001]}}}]}
    path = tmp_path / "hl-hand.yaml"
    path.write_text(yaml.safe_dump(dict(doc, groups=["P-hand", "A0"])))
    exp, cfg = load_experiment(path)
    fam = get_family(world)
    store = Store(tmp_path / "runs")
    results = run_batch(exp, cfg, store, lambda t: scripted_client("reference", t, fam),
                        sandbox_backend="local", out=lambda _: None)
    assert [r["group"] for r in results] == ["P-hand", "A0"] and all(r["passed"] for r in results)
    hand = calibration.handbook_lines(fam)
    rows = store.db.execute("SELECT test_group, checkpoint, trace_sha256 FROM episodes WHERE batch_id LIKE ?",
                            (f"{exp.name}/%",)).fetchall()
    for group, checkpoint, sha in rows:
        start = json.loads(store.get_blob("traces", sha).decode().splitlines()[0])
        assert checkpoint == 0 and start["system"] == system_prompt(True) and start["tools"] == tools(True)
        assert all((line in start["first_message"]) == (group == "P-hand") for line in hand)
        assert (f"[hand-{len(hand):02d}]" in start["first_message"]) == (group == "P-hand")

    path.write_text(yaml.safe_dump(dict(doc, phases=[{"train": doc["phases"][0]["eval"]}])))
    exp, cfg = load_experiment(path)
    assert {s.group for s in schedule(exp, cfg) if s.kind == "train"} == {"B"}
    path.write_text(yaml.safe_dump(dict(doc, family="cpu_mem", groups=["P-hand"])))
    with pytest.raises(ValidationError, match="house_lib"):
        load_experiment(path)


def test_unknown_world_ids_are_rejected():
    with pytest.raises(ValueError):
        get_family("house_lib/L16-w1")
    assert get_family("house_lib").name == "house_lib/L16-c50-w1"
    assert json.loads(get_family("house_lib/L4-c0-w2").generate(
        build_world("L4-c0-w2").lessons[0].id, "billing", 1).oracle.files["params.json"])["spec"]["table"]
