"""Cause: FIFO work queue implemented with list.pop(0) (O(n) per dequeue).

Self-contained (stdlib only): this file is copied into the judge sandbox.
"""

NAME = "queue_pop_front"
SIZES = {"small": 30, "medium": 3_000, "large": 600_000}

SYMPTOM = (
    "The {job} worker drains its backlog far too slowly once the queue of {items} gets large: "
    "throughput collapses as the backlog grows and CPU stays pegged. Processing order is part "
    "of the contract (downstream consumers depend on the checksum) and must not change."
)

MOD = (1 << 61) - 1


def gen_input(rng, n, F):
    items = [{F["key"]: f"{F['prefix']}-{i:06d}", F["value"]: rng.randint(1, 400)} for i in range(n)]
    return {F["items"]: items, "split_above": rng.randint(150, 300)}


def reference(data, F):
    from collections import deque

    limit = data["split_above"]
    q = deque(rec[F["value"]] for rec in data[F["items"]])
    h = 0
    count = 0
    total = 0
    while q:
        v = q.popleft()
        if v > limit:
            q.append(v // 2)
            q.append(v - v // 2)
            continue
        h = (h * 1_000_003 + v) % MOD
        count += 1
        total += v
    return {"processed": count, "total": total, "checksum": h}


BUGGY_CORE = '''\
"""${title} — backlog worker. Oversized ${items} are split and re-queued."""

MOD = (1 << 61) - 1


def ${verb}(data):
    limit = data["split_above"]
    queue = [rec["${value}"] for rec in data["${items}"]]
    checksum = 0
    processed = 0
    total = 0
    while queue:
        ${value} = queue.pop(0)
        if ${value} > limit:
            queue.append(${value} // 2)
            queue.append(${value} - ${value} // 2)
            continue
        checksum = (checksum * 1_000_003 + ${value}) % MOD
        processed += 1
        total += ${value}
    return {"processed": processed, "total": total, "checksum": checksum}
'''

FIXED_CORE = '''\
"""${title} — backlog worker. Oversized ${items} are split and re-queued."""

from collections import deque

MOD = (1 << 61) - 1


def ${verb}(data):
    limit = data["split_above"]
    queue = deque(rec["${value}"] for rec in data["${items}"])
    checksum = 0
    processed = 0
    total = 0
    while queue:
        ${value} = queue.popleft()
        if ${value} > limit:
            queue.append(${value} // 2)
            queue.append(${value} - ${value} // 2)
            continue
        checksum = (checksum * 1_000_003 + ${value}) % MOD
        processed += 1
        total += ${value}
    return {"processed": processed, "total": total, "checksum": checksum}
'''

# Plausible but wrong: pops from the end (LIFO), changing processing order.
WRONG_FIX_CORE = '''\
"""${title} — backlog worker. Oversized ${items} are split and re-queued."""

MOD = (1 << 61) - 1


def ${verb}(data):
    limit = data["split_above"]
    queue = [rec["${value}"] for rec in data["${items}"]]
    queue.reverse()
    checksum = 0
    processed = 0
    total = 0
    while queue:
        ${value} = queue.pop()
        if ${value} > limit:
            queue.append(${value} // 2)
            queue.append(${value} - ${value} // 2)
            continue
        checksum = (checksum * 1_000_003 + ${value}) % MOD
        processed += 1
        total += ${value}
    return {"processed": processed, "total": total, "checksum": checksum}
'''
