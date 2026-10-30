"""Input records for house_lib tasks: the sample in `examples/` and the oracle's hidden inputs.

Self-contained (stdlib only): this file is copied into the judge sandbox.
"""

from datetime import date, timedelta

REGIONS = ["eu", "us", "uk", "apac", "latam", "ca", "mena", "nordics"]
NAMES = ["ana", "bo", "chen", "dara", "eli", "femi", "gus", "hana", "ivo", "jun", "kai", "lena"]
DOMAINS = ["example.com", "mail.test", "corp.example", "inbox.test"]
SIZES = {"sample": 6, "small": 8, "medium": 40}


def gen_records(rng, n, F, region, min_cents):
    """n records in the theme's vocabulary. With n >= 2 the first record always qualifies, so a
    feature check on small inputs never degenerates to "write nothing"."""
    base = rng.randrange(10_000, 90_000)
    out = []
    for i in range(n):
        out.append({
            F["key"]: f"{F['prefix']}-{base + i:05d}",
            F["group"]: f"{F['group']}-{rng.randint(1, 9)}",
            "region": region if rng.random() < 0.6 else rng.choice(REGIONS),
            "amount_cents": rng.randint(1, 20_000),
            "due": (date(2026, 1, 1) + timedelta(days=rng.randint(0, 364))).isoformat(),
            "email": f"{rng.choice(NAMES)}.{rng.randint(1, 999)}@{rng.choice(DOMAINS)}",
        })
    if n >= 2:
        out[0]["region"] = region
        out[0]["amount_cents"] = max(out[0]["amount_cents"], min_cents)
    return out


def qualifies(rec, region, min_cents):
    return rec["region"] == region and rec["amount_cents"] >= min_cents
