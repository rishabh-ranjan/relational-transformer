"""Log-file friendly logging: plain lines, no carriage returns, no progress bars.

Every line emitted through ``log`` is a flat sequence of ``key: value`` pairs
separated by two spaces::

    eval_done_at_step: 1000  elapsed: 3m20s

Values never contain whitespace, so after stripping ANSI codes
(``re.sub(r"\\x1b\\[[0-9;]*m", "", line)``) a line parses into a dict with
``re.findall(r"(\\S+): (\\S+)", line)``. Numeric and time values are bolded,
values are column-padded, and leading indentation nests records: all cosmetic.
"""

import re

BOLD = "\033[1m"
RESET = "\033[0m"

# bold only numeric/time values: 12  3.5  0m30s  1.25GiB  2.0GiB/s -- anything
# starting with a digit (or a sign), never names, paths or free-form strings.
_NUMERIC = re.compile(r"^[+-]?\d")


def fmt_duration(secs):
    m, s = divmod(int(secs), 60)
    return f"{m}m{s:02d}s"


def fmt_bytes(n):
    return f"{n / (1024**3):.2f}GiB"


_widths: dict[str, dict[str, int]] = {}


def log(*, indent=0, **fields):
    """Emit one ``key: value`` record. Values must be whitespace-free.

    Values are padded to the widest value seen so far for that key within
    this set of keys, so repeated records of a kind line up in columns.
    """
    widths = _widths.setdefault(",".join(fields), {})
    parts = []
    for k, v in fields.items():
        v = str(v)
        widths[k] = max(widths.get(k, 0), len(v))
        v = v.ljust(widths[k])
        parts.append(
            f"{k}: {BOLD}{v}{RESET}" if _NUMERIC.match(v.strip()) else f"{k}: {v}"
        )
    print(("  " * indent + "  ".join(parts)).rstrip(), flush=True)
