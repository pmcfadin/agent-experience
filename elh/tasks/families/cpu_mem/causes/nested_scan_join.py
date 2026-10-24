"""Cause: for every record, linearly scan another collection for its match (O(n*m) join).

Self-contained (stdlib only): this file is copied into the judge sandbox.
"""

NAME = "nested_scan_join"
SIZES = {"small": 40, "medium": 1_500, "large": 80_000}

SYMPTOM = (
    "Since the {ref} table was merged in, the {job} job takes hours on the full {items} "
    "export and the host shows one core at 100%. It used to finish in seconds. Results on "
    "small samples are correct and must not change."
)


def gen_input(rng, n, F):
    n_refs = max(1, n)
    ref_keys = [f"{F['prefix']}-{rng.randrange(10**7):07d}" for _ in range(n_refs)]
    refs = []
    for k in ref_keys:
        refs.append({F["key"]: k, "rate": rng.randint(1, 9)})
    for _ in range(max(1, n_refs // 10)):
        dup = rng.choice(ref_keys)
        refs.append({F["key"]: dup, "rate": rng.randint(10, 19)})
    rng.shuffle(refs)
    groups = [f"{F['group']}-{i}" for i in range(max(1, n // 400) + 2)]
    records = []
    for _ in range(n):
        if rng.random() < 0.9:
            k = rng.choice(ref_keys)
        else:
            k = f"{F['prefix']}-x{rng.randrange(10**6):06d}"
        records.append({F["key"]: k, F["group"]: rng.choice(groups), F["value"]: rng.randint(1, 1000)})
    return {F["items"]: records, F["ref"]: refs}


def reference(data, F):
    index = {}
    for r in data[F["ref"]]:
        index.setdefault(r[F["key"]], r["rate"])
    totals = {}
    unmatched = 0
    for rec in data[F["items"]]:
        rate = index.get(rec[F["key"]])
        if rate is None:
            unmatched += 1
            continue
        g = rec[F["group"]]
        totals[g] = totals.get(g, 0) + rec[F["value"]] * rate
    return {"totals": dict(sorted(totals.items())), "unmatched": unmatched}


BUGGY_CORE = '''\
"""${title} — enrich ${items} with ${ref} rates and roll up."""


def find_rate(${ref}, ${key}):
    for row in ${ref}:
        if row["${key}"] == ${key}:
            return row["rate"]
    return None


def ${verb}(data):
    ${ref} = data["${ref}"]
    totals = {}
    unmatched = 0
    for rec in data["${items}"]:
        rate = find_rate(${ref}, rec["${key}"])
        if rate is None:
            unmatched += 1
            continue
        ${group} = rec["${group}"]
        totals[${group}] = totals.get(${group}, 0) + rec["${value}"] * rate
    return {"totals": dict(sorted(totals.items())), "unmatched": unmatched}
'''

FIXED_CORE = '''\
"""${title} — enrich ${items} with ${ref} rates and roll up."""


def build_index(${ref}):
    index = {}
    for row in ${ref}:
        index.setdefault(row["${key}"], row["rate"])
    return index


def ${verb}(data):
    index = build_index(data["${ref}"])
    totals = {}
    unmatched = 0
    for rec in data["${items}"]:
        rate = index.get(rec["${key}"])
        if rate is None:
            unmatched += 1
            continue
        ${group} = rec["${group}"]
        totals[${group}] = totals.get(${group}, 0) + rec["${value}"] * rate
    return {"totals": dict(sorted(totals.items())), "unmatched": unmatched}
'''

# Plausible but wrong: dict comprehension keeps the LAST duplicate, not the first.
WRONG_FIX_CORE = '''\
"""${title} — enrich ${items} with ${ref} rates and roll up."""


def ${verb}(data):
    index = {row["${key}"]: row["rate"] for row in data["${ref}"]}
    totals = {}
    unmatched = 0
    for rec in data["${items}"]:
        rate = index.get(rec["${key}"])
        if rate is None:
            unmatched += 1
            continue
        ${group} = rec["${group}"]
        totals[${group}] = totals.get(${group}, 0) + rec["${value}"] * rate
    return {"totals": dict(sorted(totals.items())), "unmatched": unmatched}
'''
