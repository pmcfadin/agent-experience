"""The generated in-house library `acmecore` (design §2, §4.1).

One copy per world, identical for every task in it. The API is fixed; what varies by world is
which lessons are stated and where: `source` lessons appear as bare lookup tables in the module
that implements their API, `doc` lessons as prose in `acmecore/docs/`, `hidden` lessons nowhere.

The same files are the oracle's instrumented copy: `store` keeps a call log, `clock` can be frozen
and `flags` switched off through module-level state that only the oracle's harness sets.
"""

from __future__ import annotations

import hashlib
import json
from string import Template

INIT_PY = '''\
"""acmecore: shared building blocks for Acme services.

Modules: store, audit, clock, flags, formats, ids, pii, units, retry, paging, cache, log.
See acmecore/docs/ for usage and house conventions.
"""
'''

STORE_PY = '''\
"""Row store used by every service export. Writes go to an in-process log in this build."""

_LOG = []


def write(table, row):
    """Persist one row (a dict of JSON values) to `table`."""
    _LOG.append(["write", table, dict(row)])


def rows(table):
    """Rows written to `table`, in write order."""
    return [e[2] for e in _LOG if e[0] == "write" and e[1] == table]


def reset():
    del _LOG[:]
'''

AUDIT_PY = '''\
"""Audit trail for writes made on behalf of partners."""

from acmecore import store

MODULE_AUDIT = ${module_audit}


def record(table, key):
    """Record that the row `key` of `table` was exported."""
    store._LOG.append(["audit", table, key])
'''

CLOCK_PY = '''\
"""Time sources. All return ISO 8601 strings in UTC."""

from datetime import datetime, timezone

MODULE_CLOCKS = ${module_clocks}

_FROZEN = {}


def _now():
    return datetime.now(timezone.utc)


def now():
    """Wall-clock time."""
    return _FROZEN.get("now") or _now().isoformat(timespec="seconds")


def batch_time():
    """Start of today's batch window (02:00 UTC)."""
    return _FROZEN.get("batch_time") or _now().replace(hour=2, minute=0, second=0, microsecond=0).isoformat()


def cutoff():
    """Today's settlement cutoff (17:00 UTC)."""
    return _FROZEN.get("cutoff") or _now().replace(hour=17, minute=0, second=0, microsecond=0).isoformat()


def settled_at():
    """Settlement time: wall-clock time truncated to the hour."""
    return _FROZEN.get("settled_at") or _now().replace(minute=0, second=0, microsecond=0).isoformat()
'''

FLAGS_PY = '''\
"""Feature flags. Unknown flags are on."""

MODULE_FLAGS = ${module_flags}

_OFF = set()


def on(name, tenant=None):
    """True if flag `name` is enabled for `tenant`."""
    return name not in _OFF
'''

FORMATS_PY = '''\
"""Date formatting for partner feeds."""

from datetime import date, datetime, timezone

PARTNER_STYLES = ${partner_styles}

STYLES = ("iso", "dmy_dot", "ymd_slash", "epoch_ms", "mdy_dash", "yyyymmdd")


def format_date(iso_date, style):
    """Format an ISO date string (YYYY-MM-DD) in one of STYLES."""
    d = date.fromisoformat(iso_date)
    if style == "iso":
        return d.isoformat()
    if style == "dmy_dot":
        return d.strftime("%d.%m.%Y")
    if style == "ymd_slash":
        return d.strftime("%Y/%m/%d")
    if style == "mdy_dash":
        return d.strftime("%m-%d-%Y")
    if style == "yyyymmdd":
        return d.strftime("%Y%m%d")
    if style == "epoch_ms":
        return int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp() * 1000)
    raise ValueError(f"unknown date style {style!r}")
'''

IDS_PY = '''\
"""Identifiers."""

import hashlib

MODULE_IDEM_PARTS = ${module_idem_parts}


def idem(*parts):
    """Idempotency key from the given parts (order matters)."""
    return hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()[:20]
'''

PII_PY = '''\
"""Personal data helpers."""

import hashlib

REGION_CONTACT = ${region_contact}


def mask(email):
    """First letter, then *** up to the domain: a***@example.com."""
    local, _, domain = email.partition("@")
    return f"{local[:1]}***@{domain}"


def digest(email):
    """Stable pseudonym for an email address."""
    return hashlib.sha256(email.lower().encode()).hexdigest()[:16]


def domain(email):
    """The domain part only."""
    return email.partition("@")[2]
'''

UNITS_PY = '''\
"""Money representations. Amounts are held as integer cents internally."""

MODULE_UNITS = ${module_units}

UNITS = ("cents", "decimal_str", "milli", "cents_str")


def convert(cents, unit):
    """Represent an integer amount of cents in one of UNITS."""
    if unit == "cents":
        return int(cents)
    if unit == "decimal_str":
        return f"{cents // 100}.{cents % 100:02d}"
    if unit == "milli":
        return int(cents) * 10
    if unit == "cents_str":
        return str(int(cents))
    raise ValueError(f"unknown unit {unit!r}")
'''

# Lesson-neutral modules: realistic surface area the agent may read but no lesson depends on.
RETRY_PY = '''\
"""Retrying calls that fail transiently."""

import time


class TransientError(Exception):
    pass


def call(fn, *args, attempts=3, delay_s=0.0, **kwargs):
    """Call fn, retrying TransientError up to `attempts` times."""
    for i in range(attempts):
        try:
            return fn(*args, **kwargs)
        except TransientError:
            if i == attempts - 1:
                raise
            time.sleep(delay_s)
'''

PAGING_PY = '''\
"""Paging over in-memory sequences."""


def pages(items, size):
    """Yield consecutive slices of `items` of at most `size` elements."""
    for i in range(0, len(items), size):
        yield items[i:i + size]
'''

CACHE_PY = '''\
"""A tiny process-local cache."""

_DATA = {}


def get(key, default=None):
    return _DATA.get(key, default)


def put(key, value):
    _DATA[key] = value


def clear():
    _DATA.clear()
'''

LOG_PY = '''\
"""Structured logging to stderr."""

import json
import sys


def info(event, **fields):
    print(json.dumps({"level": "info", "event": event, **fields}, sort_keys=True), file=sys.stderr)
'''

API_DOC = '''\
# acmecore

Shared building blocks for Acme services. Every service export writes its rows through
`acmecore.store.write(table, row)`; tests read them back with `store.rows(table)`.

| Module | API |
|---|---|
| `store` | `write(table, row)`, `rows(table)`, `reset()` |
| `audit` | `record(table, key)` |
| `clock` | `now()`, `batch_time()`, `cutoff()`, `settled_at()` (ISO 8601, UTC) |
| `flags` | `on(name, tenant=None)` |
| `formats` | `format_date(iso_date, style)`; styles `iso`, `dmy_dot`, `ymd_slash`, `epoch_ms`, `mdy_dash`, `yyyymmdd` |
| `ids` | `idem(*parts)` |
| `pii` | `mask(email)`, `digest(email)`, `domain(email)` |
| `units` | `convert(cents, unit)`; units `cents`, `decimal_str`, `milli`, `cents_str` |
| `retry` | `call(fn, *args, attempts=3)`, `TransientError` |
| `paging` | `pages(items, size)` |
| `cache` | `get(key, default)`, `put(key, value)`, `clear()` |
| `log` | `info(event, **fields)` |

Topic notes: `partners.md`, `privacy.md`, `money.md`, `time.md`, `events.md`, `audit.md`, `rollout.md`.
'''

# Topic docs: an always-present lesson-neutral intro, then the world's `doc` lessons as prose.
TOPIC_DOCS = {
    "partners.md": ("Partner feeds", "Partner feeds are CSV/JSON exports written to a partner's table. "
                    "Dates are formatted with `formats.format_date`."),
    "privacy.md": ("Privacy", "Customer contact details are personal data. `acmecore.pii` has the helpers."),
    "money.md": ("Money", "Amounts are integer cents inside services; `units.convert` changes the representation."),
    "time.md": ("Time", "Use `acmecore.clock` for timestamps so batch runs are reproducible."),
    "events.md": ("Events and idempotency", "Rows that may be replayed carry an idempotency key from `ids.idem`."),
    "audit.md": ("Audit", "Partner exports are audited with `audit.record(table, key)`."),
    "rollout.md": ("Rollout", "New behaviour ships behind `acmecore.flags`."),
}

MODULE_FILES = {
    "acmecore/__init__.py": INIT_PY, "acmecore/store.py": STORE_PY, "acmecore/audit.py": AUDIT_PY,
    "acmecore/clock.py": CLOCK_PY, "acmecore/flags.py": FLAGS_PY, "acmecore/formats.py": FORMATS_PY,
    "acmecore/ids.py": IDS_PY, "acmecore/pii.py": PII_PY, "acmecore/units.py": UNITS_PY,
    "acmecore/retry.py": RETRY_PY, "acmecore/paging.py": PAGING_PY, "acmecore/cache.py": CACHE_PY,
    "acmecore/log.py": LOG_PY,
}


def _table(entries: dict) -> str:
    if not entries:
        return "{}"
    body = "".join(f"    {k!r}: {v!r},\n" for k, v in sorted(entries.items()))
    return "{\n" + body + "}"


def build_library(source_tables: dict[str, dict], doc_lines: dict[str, list[str]]) -> dict[str, str]:
    """acmecore files for one world. `source_tables` maps each table placeholder (e.g.
    `partner_styles`) to its entries; `doc_lines` maps a topic doc to its convention sentences."""
    subs = {k: _table(v) for k, v in source_tables.items()}
    files = {path: Template(src).substitute(subs) for path, src in MODULE_FILES.items()}
    files["acmecore/docs/README.md"] = API_DOC
    for name, (title, intro) in TOPIC_DOCS.items():
        lines = doc_lines.get(name, [])
        conv = ("\n## House conventions\n\n" + "".join(f"- {s}\n" for s in lines)) if lines else ""
        files[f"acmecore/docs/{name}"] = f"# {title}\n\n{intro}\n{conv}"
    return files


def fingerprint(files: dict[str, str]) -> str:
    blob = json.dumps({k: v for k, v in files.items() if k.startswith("acmecore/")}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()
