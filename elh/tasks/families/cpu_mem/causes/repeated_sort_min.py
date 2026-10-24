"""Cause: re-sorting the whole pending list to take the minimum on every iteration.

Self-contained (stdlib only): this file is copied into the judge sandbox.
"""

NAME = "repeated_sort_min"
SIZES = {"small": 30, "medium": 1_500, "large": 30_000}

SYMPTOM = (
    "The {job} scheduler that orders {items} by priority has become the slowest stage in the "
    "pipeline; with a full day's backlog it runs far past its deadline at 100% CPU. The dispatch "
    "order (priority, then id) is relied on downstream and must stay identical."
)

MOD = (1 << 61) - 1


def gen_input(rng, n, F):
    ids = rng.sample(range(10**6), n)
    items = []
    for i in ids:
        items.append({
            F["key"]: f"{F['prefix']}-{i:06d}",
            "priority": rng.randint(0, max(3, n // 50)),
            "repeats": rng.choice([0, 0, 1, 2, 3]),
            "backoff": rng.randint(1, 7),
        })
    return {F["items"]: items}


def _num(key):
    return int(key.rsplit("-", 1)[1])


def reference(data, F):
    import heapq

    heap = [(r["priority"], r[F["key"]], r["repeats"], r["backoff"]) for r in data[F["items"]]]
    heapq.heapify(heap)
    h = 0
    dispatched = 0
    last_priority = None
    while heap:
        p, k, reps, back = heapq.heappop(heap)
        h = (h * 1_000_003 + _num(k)) % MOD
        dispatched += 1
        last_priority = p
        if reps > 0:
            heapq.heappush(heap, (p + back, k, reps - 1, back))
    return {"dispatched": dispatched, "last_priority": last_priority, "checksum": h}


BUGGY_CORE = '''\
"""${title} — priority dispatcher for ${items}. Lowest priority value goes first; ties by id."""

from .util import id_number

MOD = (1 << 61) - 1


def ${verb}(data):
    pending = [(r["priority"], r["${key}"], r["repeats"], r["backoff"]) for r in data["${items}"]]
    checksum = 0
    dispatched = 0
    last_priority = None
    while pending:
        pending.sort()
        priority, ${key}, repeats, backoff = pending.pop(0)
        checksum = (checksum * 1_000_003 + id_number(${key})) % MOD
        dispatched += 1
        last_priority = priority
        if repeats > 0:
            pending.append((priority + backoff, ${key}, repeats - 1, backoff))
    return {"dispatched": dispatched, "last_priority": last_priority, "checksum": checksum}
'''

FIXED_CORE = '''\
"""${title} — priority dispatcher for ${items}. Lowest priority value goes first; ties by id."""

import heapq

from .util import id_number

MOD = (1 << 61) - 1


def ${verb}(data):
    pending = [(r["priority"], r["${key}"], r["repeats"], r["backoff"]) for r in data["${items}"]]
    heapq.heapify(pending)
    checksum = 0
    dispatched = 0
    last_priority = None
    while pending:
        priority, ${key}, repeats, backoff = heapq.heappop(pending)
        checksum = (checksum * 1_000_003 + id_number(${key})) % MOD
        dispatched += 1
        last_priority = priority
        if repeats > 0:
            heapq.heappush(pending, (priority + backoff, ${key}, repeats - 1, backoff))
    return {"dispatched": dispatched, "last_priority": last_priority, "checksum": checksum}
'''

# Plausible but wrong: heap keyed on (priority, insertion counter) breaks ties by arrival, not id.
WRONG_FIX_CORE = '''\
"""${title} — priority dispatcher for ${items}. Lowest priority value goes first; ties by id."""

import heapq
import itertools

from .util import id_number

MOD = (1 << 61) - 1


def ${verb}(data):
    counter = itertools.count()
    pending = [(r["priority"], next(counter), r["${key}"], r["repeats"], r["backoff"]) for r in data["${items}"]]
    heapq.heapify(pending)
    checksum = 0
    dispatched = 0
    last_priority = None
    while pending:
        priority, _, ${key}, repeats, backoff = heapq.heappop(pending)
        checksum = (checksum * 1_000_003 + id_number(${key})) % MOD
        dispatched += 1
        last_priority = priority
        if repeats > 0:
            heapq.heappush(pending, (priority + backoff, next(counter), ${key}, repeats - 1, backoff))
    return {"dispatched": dispatched, "last_priority": last_priority, "checksum": checksum}
'''
