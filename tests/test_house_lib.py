"""house_lib task family (docs/design/house-lib-family.md): worlds, tasks, oracle, validation."""

import json
from collections import Counter
from pathlib import Path

import pytest
import yaml

from elh import batch
from elh.batch import load_experiment, run_batch, schedule
from elh.controller import regrade_episode, run_single_episode
from elh.judge import grade
from elh.registry import load_config
from elh.runner.scripts import scripted_client
from elh.sandbox.local import LocalSandbox
from elh.store import Store
from elh.tasks.base import get_family
from elh.tasks.families.house_lib import calibration
from elh.tasks.families.house_lib.lessons import MODULE_WIDE, TEMPLATES, build_world, gold_text
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
    res = {r.lesson: r for r in calibration.filter_lessons(eps, ["x", "y", "z"], told="P-hand")}
    assert not res["x"].keep and "guessable" in res["x"].reason          # A passes 30%
    assert not res["y"].keep and "not learnable" in res["y"].reason      # P-hand passes 0%
    assert res["z"].keep and "undecided" in res["z"].reason


def test_p_task_shows_only_the_tasks_own_handbook_lines(tmp_path):
    world = "house_lib/L16-c0-w1"
    fam = get_family(world)
    path = tmp_path / "hl-task.yaml"
    path.write_text(yaml.safe_dump({"name": "hl-task", "base_config": str(CONFIG), "family": world, "groups": ["P-task"],
                                    "phases": [{"eval": {"tasks": {"items": [[fam.causes[0], "billing", 3000]]}}}]}))
    exp, cfg = load_experiment(path)
    mem = batch.static_memory(Store(tmp_path / "runs"), exp, 0, "P-task")
    gold = dict(zip([les.id for les in fam.world.lessons], calibration.handbook_lines(fam)))
    for les in fam.causes:
        task = fam.generate(les, "billing", 3000)
        assert [it.text for it in mem.recall(task, 5)] == [gold[x] for x in fam.causes if x in task.signature["lessons"]]
    path.write_text(path.read_text().replace(world, "cpu_mem").replace(fam.causes[0], "queue_pop_front"))
    with pytest.raises(Exception, match="house_lib"):
        load_experiment(path)


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


@pytest.mark.parametrize("path", sorted((Path(__file__).parent.parent / "experiments" / "batches").glob("exp*-calib-*.yaml")),
                         ids=lambda p: p.stem)
def test_calibration_batches_are_the_calibration_panel(path):
    exp, _ = load_experiment(path)
    fam = get_family(exp.family)
    if "-rest" in path.stem:  # remainder of an interrupted batch: one group, a subset of the original's panel
        whole, _ = load_experiment(path.parent / (path.stem.split("-rest")[0] + ".yaml"))
        assert len(exp.groups) == 1 and exp.groups[0] in whole.groups and len(exp.phases) == 1
        assert (exp.family, exp.base_config, exp.overrides, exp.histories) == \
            (whole.family, whole.base_config, whole.overrides, whole.histories)
        assert exp.phases[0].spec.label == whole.phases[0].spec.label
        panel = whole.phases[0].spec.tasks.triples()
        assert set(exp.phases[0].spec.tasks.triples()) <= set(panel)
        return
    assert exp.groups == ["A", "P-hand"] and len(exp.phases) == 1
    learn = path.parent / path.name.replace("exp21-calib-", "exp24-learn-").replace("exp40-calib-", "exp41-learn-")
    if path.stem.startswith(("exp21-", "exp40-")):  # its learnability control runs the same panel
        ctl, _ = load_experiment(learn)
        assert ctl.groups == ["P-task"] and ctl.phases[0].spec.tasks.triples() == exp.phases[0].spec.tasks.triples()
    per_lesson = 8 if path.stem.startswith(("exp21-", "exp40-")) else 4  # exp21 doubled the panel (design §10)
    assert exp.phases[0].spec.tasks.triples() == calibration.calibration_tasks(fam, per_lesson=per_lesson)


def test_ticket_names_key_and_group_columns_and_no_identity_unit():
    fam = get_family("house_lib/L16-c50-w1")
    for les in fam.world.lessons:
        task = fam.generate(les.id, "logistics", 3000)
        spec = fam.spec(task)
        assert f"`{spec['key']}`: the record's {spec['key']} (the row's key column)" in task.prompt
        assert f"`{spec['group']}`: the record's {spec['group']} (the row's group column)" in task.prompt
    assert "cents" not in TEMPLATES["unit_convention"].choices


def test_handbook_in_ticket_variant_changes_only_the_ticket(family):
    plain, t = get_family("house_lib/L4-c0-w1"), get_family("house_lib/L4-c0-w1+t")
    for cause in plain.causes:
        a, b = plain.generate(cause, "telemetry", 3005), t.generate(cause, "telemetry", 3005)
        assert a.oracle.files == b.oracle.files and a.signature == b.signature and a.protected == b.protected
        assert {k for k in a.workspace if a.workspace[k] != b.workspace.get(k)} == {"TICKET.md"}
        assert a.prompt != b.prompt and all(line in b.prompt for line in calibration.handbook_lines(plain))
    with pytest.raises(ValueError, match="variant"):
        get_family("house_lib/L4-c0-w1+x")


def test_unknown_world_ids_are_rejected():
    with pytest.raises(ValueError):
        get_family("house_lib/L16-w1")
    assert get_family("house_lib").name == "house_lib/L16-c50-w1"
    assert json.loads(get_family("house_lib/L4-c0-w2").generate(
        build_world("L4-c0-w2").lessons[0].id, "billing", 1).oracle.files["params.json"])["spec"]["table"]


def test_experience_block_header_and_position(family):
    from elh.runner.prompts import EXPERIENCE_HEADERS, first_user_message

    task = family.generate(family.causes[0], "billing", 1)
    default = first_user_message(task, "/ws", "[x]\nnote", memory_check=True)
    assert default == first_user_message(task, "/ws", "[x]\nnote", memory_check=True,
                                         header="hedged", position="before_ticket")
    assert default.index("<experience>") < default.index("<ticket>") and EXPERIENCE_HEADERS["hedged"] in default
    plain = first_user_message(task, "/ws", "[x]\nnote", header="plain")
    assert EXPERIENCE_HEADERS["plain"] in plain and "may or may not" not in plain
    after = first_user_message(task, "/ws", "[x]\nnote", memory_check=True, position="after_ticket")
    assert after.index("</ticket>") < after.index("<experience>") < after.index("<memory_check>")
    assert first_user_message(task, "/ws", "", position="after_ticket") == first_user_message(task, "/ws", "")
    with pytest.raises(ValueError, match="position"):
        first_user_message(task, "/ws", "[x]", position="sideways")


def test_grid_panels_and_scores_over_kept_lessons(tmp_path):
    from elh.tasks.families.house_lib import grid

    fam = get_family("house_lib/L16-c0-w1")
    kept = [c for c in fam.causes if c not in ("feature_flag.ledger", "partner_format.fjord")]
    train, evals = grid.grid_tasks(fam, kept)
    assert Counter(c for c, _, _ in train) == {c: 2 for c in fam.causes} and len(train) == 32
    assert len(evals) == grid.EVAL_TASKS and {c for c, _, _ in evals} == set(kept)
    assert max(Counter(c for c, _, _ in evals).values()) - min(Counter(c for c, _, _ in evals).values()) <= 1
    assert all(s < grid.EVAL_SEED for *_, s in train) and all(s >= grid.EVAL_SEED for *_, s in evals)
    assert grid.grid_tasks(fam, kept) == (train, evals)
    checks = {"integrity": True, "public_tests": True, "feature": True,
              "lesson:feature_flag.ledger": False, "lesson:unit_convention.ledger": True}
    assert grid.scored_pass(checks, kept) and "lesson:feature_flag.ledger" not in grid.scored_checks(checks, kept)
    assert not grid.scored_pass(dict(checks, feature=False), kept)
    assert not grid.scored_pass(dict(checks, **{"lesson:unit_convention.ledger": False}), kept)
    with pytest.raises(ValueError):
        grid.grid_tasks(fam, ["no.such"])
    path = tmp_path / "w.json"
    path.write_text(json.dumps({"world": "L16-c0-w1", "lessons": [{"lesson": c, "keep": c in kept} for c in fam.causes]}))
    assert grid.kept_lessons("L16-c0-w1", path) == kept
    with pytest.raises(ValueError):
        grid.kept_lessons("L4-c0-w1", path)
    learn, static = grid.batch_docs(fam, kept)
    for doc in (learn, static):
        doc["base_config"] = str(Path(__file__).parent.parent / doc["base_config"])
        (tmp_path / f"{doc['name']}.yaml").write_text(yaml.safe_dump(doc))
        exp, cfg = load_experiment(tmp_path / f"{doc['name']}.yaml")
        steps = schedule(exp, cfg)
        assert sorted((s.task.cause, s.task.theme, s.task.seed) for s in steps if s.kind == "eval" and s.history == 0
                      and s.group == exp.groups[0]) == sorted(evals)
    assert static["static"] == {"from_experiment": learn["name"], "group": "B", "after": 32}
    tuned, _ = grid.batch_docs(fam, kept, recall=grid.GRID_RECALL)
    o = tuned["overrides"]
    assert (o["memory_service.k"], o["rules"]["recall_k"], o["memory_service.recall_max_tokens"]) == grid.GRID_RECALL
    assert grid.GRID_RECALL in grid.TUNE_SETTINGS and "memory_service.k" not in learn["overrides"]


@pytest.mark.parametrize("path", sorted((Path(__file__).parent.parent / "experiments" / "batches").glob("exp2[23]-grid-*.yaml")),
                         ids=lambda p: p.stem)
def test_grid_batches_match_the_generator(path):
    from elh.tasks.families.house_lib import grid

    doc = yaml.safe_load(path.read_text())
    world = doc["family"].split("/")[1]
    learn, static = grid.batch_docs(get_family(doc["family"]), grid.kept_lessons(world))
    assert doc == (learn if doc["name"].startswith("exp22-") else static)


@pytest.mark.parametrize("path", sorted((Path(__file__).parent.parent / "experiments" / "batches").glob("exp28-tune-*.yaml")),
                         ids=lambda p: p.stem)
def test_tune_batches_match_the_generator(path):
    from elh.tasks.families.house_lib import grid

    doc = yaml.safe_load(path.read_text())
    world = doc["family"].split("/")[1]
    assert doc in grid.tune_docs(get_family(doc["family"]), grid.kept_lessons(world))
    assert all(seed >= grid.TUNE_SEED for _, _, seed in doc["phases"][0]["eval"]["tasks"]["items"])


# The sealed study's worlds (roadmap 2026-11-07): world seed 2 for all six cells.
W1_DIGEST = "c7468a261fcee779966e0629d10b2103f87a123ec377f765920daec01cdc1f41"
CELLS = [(n, c) for n in (4, 16, 64) for c in (0, 50)]


def test_w1_worlds_are_unchanged():
    """sha256 over the six w1 worlds' catalogues and handbooks and two tasks each (first and last
    lesson), taken before w2 was added: any change to w1 generation changes it."""
    import hashlib

    h = hashlib.sha256()
    for n, c in CELLS:
        fam = get_family(f"house_lib/L{n}-c{c}-w1")
        h.update(repr(fam.world).encode())
        h.update(calibration.handbook(fam).encode())
        for cause in (fam.causes[0], fam.causes[-1]):
            t = fam.generate(cause, "billing", 3000)
            h.update(json.dumps([t.task_id, t.content_hash, t.prompt], sort_keys=True).encode())
    assert h.hexdigest() == W1_DIGEST


@pytest.mark.parametrize("n,c", CELLS, ids=lambda x: str(x))
def test_w2_worlds_are_deterministic_and_differ_from_w1(n, c):
    w1, w2 = get_family(f"house_lib/L{n}-c{c}-w1"), get_family(f"house_lib/L{n}-c{c}-w2")
    again = get_family(f"house_lib/L{n}-c{c}-w2")
    assert w2.world == again.world and w2.world.seed == 2 and len(w2.world.lessons) == n
    assert calibration.handbook(w2) == calibration.handbook(again)
    assert calibration.handbook(w1) != calibration.handbook(w2)
    assert calibration.calibration_tasks(w1, 8) != calibration.calibration_tasks(w2, 8)
    cause = next((x for x in w2.causes if x in w1.causes), None)
    t, u = w2.generate(w2.causes[0], "billing", 3000), again.generate(w2.causes[0], "billing", 3000)
    assert (t.content_hash, t.prompt, t.workspace) == (u.content_hash, u.prompt, u.workspace)
    if cause:  # a lesson both worlds have: still a different task (own records, library and oracle)
        assert w1.generate(cause, "billing", 3000).content_hash != w2.generate(cause, "billing", 3000).content_hash


def test_w2_batch_files_are_the_generated_ones():
    batches = Path(__file__).parent.parent / "experiments" / "batches"
    for n, c in CELLS:
        fam = get_family(f"house_lib/L{n}-c{c}-w2")
        for (prefix, text) in zip(calibration.batch_prefixes(fam.world.id), calibration.batch_texts(fam)):
            assert (batches / f"{prefix}-{fam.world.id}.yaml").read_text() == text


@pytest.mark.parametrize("n,c", CELLS, ids=lambda x: str(x))
def test_batch_texts_reproduce_the_w1_files(n, c):
    batches = Path(__file__).parent.parent / "experiments" / "batches"
    fam = get_family(f"house_lib/L{n}-c{c}-w1")
    for prefix, text in zip(calibration.batch_prefixes(fam.world.id), calibration.batch_texts(fam)):
        old = (batches / f"{prefix}-{fam.world.id}.yaml").read_text()
        assert text[text.index("name:"):] == old[old.index("name:"):]


BATCHES = Path(__file__).parent.parent / "experiments" / "batches"


@pytest.mark.parametrize("path", sorted(BATCHES.glob("exp29*-*.yaml")) + sorted(BATCHES.glob("exp30*-*.yaml")),
                         ids=lambda p: p.stem)
def test_stage43_batches_match_the_generator(path):
    from elh.tasks.families.house_lib import grid

    doc = yaml.safe_load(path.read_text())
    world = doc["family"].split("/")[1]
    assert doc in grid.grid43_docs(get_family(doc["family"]), grid.kept_lessons(world))


def test_stage43_batches_exist_for_every_calibrated_world():
    """A world whose verdict has been written must have its 4.3 files: `python -m ...house_lib.grid` makes them."""
    from elh.tasks.families.house_lib import grid

    want = set()
    for world in grid.WORLDS_43:
        if (grid.CALIBRATION_DIR / f"{world}.json").exists():
            want |= {d["name"] for d in grid.grid43_docs(get_family(f"house_lib/{world}"), grid.kept_lessons(world))}
    assert want and want == {p.stem for p in BATCHES.glob("exp29*-*.yaml")} | {p.stem for p in BATCHES.glob("exp30*-*.yaml")}


def test_stage43_l16_reuses_4_1_and_l64_splits_history_0_as_the_gate(tmp_path):
    from elh.tasks.families.house_lib import grid

    l16 = get_family("house_lib/L16-c50-w1")
    learn, static = grid.grid43_docs(l16, grid.kept_lessons("L16-c50-w1"))
    assert (learn["name"], static["name"]) == ("exp29-grid-L16-c50-w1", "exp30-grid-static-L16-c50-w1")
    assert learn["groups"] == ["B"] and static["groups"] == ["P-all", "P-all-ep", "P-oracle"]  # 4.1's A, P-hand reused
    assert "history_start" not in learn and learn["histories"] == 3
    l4, _ = grid.grid43_docs(get_family("house_lib/L4-c0-w1"), grid.kept_lessons("L4-c0-w1"))
    assert l4["groups"] == ["A", "B", "P-hand"]
    for doc in (learn, static, l4):  # B recalls k14r12 in training and eval, in every world
        o = doc["overrides"]
        assert (o["memory_service.k"], o["rules"]["recall_k"], o["memory_service.recall_max_tokens"]) == grid.GRID_RECALL
    old = yaml.safe_load((BATCHES / "exp22-grid-L16-c50-w1.yaml").read_text())
    assert learn["phases"] == old["phases"] and learn["family"] == old["family"]  # same tickets as 4.1's A and P-hand

    # L64 needs a calibration verdict; use a synthetic one (never written to the repository).
    fam = get_family("house_lib/L64-c50-w1")
    kept = [c for c in fam.causes if c != fam.causes[0]]
    gate, gate_static, rest, rest_static = grid.grid43_docs(fam, kept)
    assert [d["name"] for d in (gate, gate_static, rest, rest_static)] == [
        "exp29b-gate-L64-c50-w1", "exp30b-gate-static-L64-c50-w1", "exp29-grid-L64-c50-w1", "exp30-grid-static-L64-c50-w1"]
    assert (gate["histories"], rest["histories"], rest["history_start"]) == (1, 2, 1) and "history_start" not in gate
    assert gate["groups"] == rest["groups"] == ["A", "B", "P-hand"]
    assert gate_static["static"] == {"from_experiment": gate["name"], "group": "B", "after": 128}
    assert rest_static["static"] == {"from_experiment": rest["name"], "group": "B", "after": 128}
    assert gate_static["histories"] == 1 and rest_static["history_start"] == 1
    whole = grid.batch_docs(fam, kept, recall=grid.GRID_RECALL)[0]
    assert gate["replicates"] == rest["replicates"] == {"B": 3}
    assert gate_static["replicates"] == rest_static["replicates"] == {"P-all-ep": 3}
    whole["replicates"] = gate["replicates"]
    assert gate["phases"] == rest["phases"] == whole["phases"]
    # the split runs the histories exactly as one batch of three would (same tickets, orders, checkpoints)
    steps = {}
    for doc in (gate, rest, whole):
        doc["base_config"] = str(CONFIG)
        path = tmp_path / f"{doc['name']}.yaml"
        path.write_text(yaml.safe_dump(doc))
        exp, cfg = load_experiment(path)
        steps[doc["name"]] = [(s.history, s.phase, s.kind, s.group, s.task.task_id, s.after, s.replicate) for s in schedule(exp, cfg)]
    assert steps[gate["name"]] + steps[rest["name"]] == steps[whole["name"]]
    assert sum(k == "train" for _, _, k, *_ in steps[gate["name"]]) == 128
    assert {h for h, *_ in steps[gate["name"]]} == {0} and {h for h, *_ in steps[rest["name"]]} == {1, 2}


def test_stage43_replicates_only_at_the_primary_cell(tmp_path):
    from collections import Counter

    from elh.tasks.families.house_lib import grid

    def counts(doc):
        doc = dict(doc, base_config=str(CONFIG))
        path = tmp_path / f"{doc['name']}.yaml"
        path.write_text(yaml.safe_dump(doc))
        exp, cfg = load_experiment(path)
        steps = schedule(exp, cfg)
        ev = [s for s in steps if s.kind == "eval"]
        per = Counter((s.history, s.group, s.task.task_id) for s in ev)
        ids = Counter((s.history, s.group, s.task.task_id, s.replicate) for s in ev)
        assert max(ids.values()) == 1  # replicate ids distinguish every episode
        assert all(s.replicate == 0 for s in steps if s.kind == "train")
        return len(steps), per, {s.replicate for s in ev}

    for world, total in (("L64-c50-w1", 816 + 288), ("L64-c0-w1", 816)):
        fam = get_family(f"house_lib/{world}")
        docs = grid.grid43_docs(fam, list(fam.causes))
        n = 0
        for doc in docs:
            size, per, reps = counts(doc)
            n += size
            for (_, group, _), k in per.items():
                assert k == (3 if world == grid.REPLICATE_WORLD and group in ("B", "P-all-ep") else 1)
            assert reps == ({0, 1, 2} if "replicates" in doc else {0})
        assert n == total
    for world in ("L4-c50-w1", "L16-c50-w1"):  # no other world has replicates
        assert all("replicates" not in d for d in grid.grid43_docs(get_family(f"house_lib/{world}"),
                                                                  grid.kept_lessons(world)))
