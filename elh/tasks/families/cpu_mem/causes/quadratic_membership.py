"""Cause: membership tests against a list inside a loop (O(n*m)).

Self-contained (stdlib only): this file is copied into the judge sandbox.
"""

NAME = "quadratic_membership"
SIZES = {"small": 40, "medium": 2_000, "large": 200_000}

SYMPTOM = (
    "The {job} step has slowed to a crawl as our {items} volume grew. On a production-sized "
    "export it pins one CPU core and never finishes inside the batch window. Output on small "
    "inputs is correct and must stay exactly the same (including order)."
)


def gen_input(rng, n, F):
    n_keys = max(1, n // 2)
    keys = [f"{F['prefix']}-{rng.randrange(10**7):07d}" for _ in range(n_keys)]
    groups = [f"{F['group']}-{i}" for i in range(max(1, n // 500) + 2)]
    records = []
    for _ in range(n):
        records.append({
            F["key"]: rng.choice(keys),
            F["group"]: rng.choice(groups),
            F["value"]: rng.randint(-5, 500),
        })
    blocked = rng.sample(keys, k=min(len(keys), max(0, n // 3)))
    return {F["items"]: records, "blocked": blocked}


def reference(data, F):
    blocked = set(data["blocked"])
    seen = set()
    order = []
    totals = {}
    for rec in data[F["items"]]:
        k = rec[F["key"]]
        if k in blocked or rec[F["value"]] <= 0:
            continue
        g = rec[F["group"]]
        totals[g] = totals.get(g, 0) + rec[F["value"]]
        if k not in seen:
            seen.add(k)
            order.append(k)
    return {"unique": order, "totals": dict(sorted(totals.items()))}


BUGGY_CORE = '''\
"""${title} — de-duplication and roll-up stage."""

from .util import is_valid


def ${verb}(data):
    ${items} = data["${items}"]
    blocked = list(data["blocked"])
    seen = []
    totals = {}
    for rec in ${items}:
        if not is_valid(rec, "${value}"):
            continue
        ${key} = rec["${key}"]
        if ${key} in blocked:
            continue
        ${group} = rec["${group}"]
        totals[${group}] = totals.get(${group}, 0) + rec["${value}"]
        if ${key} not in seen:
            seen.append(${key})
    return {"unique": seen, "totals": dict(sorted(totals.items()))}
'''

FIXED_CORE = '''\
"""${title} — de-duplication and roll-up stage."""

from .util import is_valid


def ${verb}(data):
    ${items} = data["${items}"]
    blocked = set(data["blocked"])
    seen = set()
    order = []
    totals = {}
    for rec in ${items}:
        if not is_valid(rec, "${value}"):
            continue
        ${key} = rec["${key}"]
        if ${key} in blocked:
            continue
        ${group} = rec["${group}"]
        totals[${group}] = totals.get(${group}, 0) + rec["${value}"]
        if ${key} not in seen:
            seen.add(${key})
            order.append(${key})
    return {"unique": order, "totals": dict(sorted(totals.items()))}
'''

# Plausible but wrong: fast, but loses first-seen order.
WRONG_FIX_CORE = '''\
"""${title} — de-duplication and roll-up stage."""

from .util import is_valid


def ${verb}(data):
    ${items} = data["${items}"]
    blocked = set(data["blocked"])
    seen = set()
    totals = {}
    for rec in ${items}:
        if not is_valid(rec, "${value}"):
            continue
        ${key} = rec["${key}"]
        if ${key} in blocked:
            continue
        ${group} = rec["${group}"]
        totals[${group}] = totals.get(${group}, 0) + rec["${value}"]
        seen.add(${key})
    return {"unique": sorted(seen), "totals": dict(sorted(totals.items()))}
'''
