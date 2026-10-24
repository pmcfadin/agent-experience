"""Cause: naive recursion with overlapping subproblems (exponential time).

Self-contained (stdlib only): this file is copied into the judge sandbox.
"""

NAME = "exponential_recursion"
SIZES = {"small": 18, "medium": 45, "large": 2_500}

SYMPTOM = (
    "The {job} planner that counts how many distinct ways each order of {items} can be packed "
    "into standard bundle sizes works for tiny orders but hangs on realistic order sizes "
    "(one core at 100%, never returns). Counts are reported modulo 1,000,000,007 and must "
    "stay exactly the same."
)

MOD = 1_000_000_007


def gen_input(rng, n, F):
    sizes = [1] + sorted(rng.sample(range(2, 9), rng.randint(3, 4)))
    n_orders = 3 if n > 100 else 6
    orders = [rng.randint(max(0, n // 2), n) for _ in range(n_orders)]
    if n <= 20:
        orders.append(0)
    return {"bundle_sizes": sizes, "orders": orders}


def reference(data, F):
    sizes = data["bundle_sizes"]
    results = []
    for target in data["orders"]:
        ways = [0] * (target + 1)
        ways[0] = 1
        for s in sizes:
            for t in range(s, target + 1):
                ways[t] = (ways[t] + ways[t - s]) % MOD
        results.append(ways[target])
    return {"ways": results}


BUGGY_CORE = '''\
"""${title} — count distinct bundle combinations per order (order of bundles irrelevant)."""

import sys

MOD = 1_000_000_007

sys.setrecursionlimit(20_000)  # large orders recurse deeply


def count_ways(sizes, i, remaining):
    if remaining == 0:
        return 1
    if remaining < 0 or i == len(sizes):
        return 0
    use_it = count_ways(sizes, i, remaining - sizes[i])
    skip_it = count_ways(sizes, i + 1, remaining)
    return (use_it + skip_it) % MOD


def ${verb}(data):
    sizes = data["bundle_sizes"]
    return {"ways": [count_ways(sizes, 0, target) for target in data["orders"]]}
'''

FIXED_CORE = '''\
"""${title} — count distinct bundle combinations per order (order of bundles irrelevant)."""

MOD = 1_000_000_007


def count_ways(sizes, target):
    ways = [0] * (target + 1)
    ways[0] = 1
    for s in sizes:
        for t in range(s, target + 1):
            ways[t] = (ways[t] + ways[t - s]) % MOD
    return ways[target]


def ${verb}(data):
    sizes = data["bundle_sizes"]
    return {"ways": [count_ways(sizes, target) for target in data["orders"]]}
'''

# Plausible but wrong: iterates targets in the outer loop, which counts ordered sequences
# (permutations) instead of combinations.
WRONG_FIX_CORE = '''\
"""${title} — count distinct bundle combinations per order (order of bundles irrelevant)."""

MOD = 1_000_000_007


def count_ways(sizes, target):
    ways = [0] * (target + 1)
    ways[0] = 1
    for t in range(1, target + 1):
        for s in sizes:
            if s <= t:
                ways[t] = (ways[t] + ways[t - s]) % MOD
    return ways[target]


def ${verb}(data):
    sizes = data["bundle_sizes"]
    return {"ways": [count_ways(sizes, target) for target in data["orders"]]}
'''
