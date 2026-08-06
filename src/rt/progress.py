"""Log-file friendly progress reporting: plain lines, no carriage returns.

Every line this module emits -- and every line the callers below emit through
``log`` -- is a flat sequence of ``key: value`` pairs separated by two spaces::

    progress: eval@1000  n: 12  total: 57  pct: 21  elapsed: 0m30s

Values never contain whitespace, so after stripping ANSI codes
(``re.sub(r"\\x1b\\[[0-9;]*m", "", line)``) a line parses into a dict with
``re.findall(r"(\\S+): (\\S+)", line)``. Numeric and time values are bolded,
values are column-padded, and leading indentation nests records: all cosmetic.
"""

import re
import time

BOLD = "\033[1m"
RESET = "\033[0m"

# bold only numeric/time values: 12  3.5  0m30s  1.25GiB  2.0GiB/s -- anything
# starting with a digit (or a sign), never names, paths or free-form strings.
_NUMERIC = re.compile(r"^[+-]?\d")


def fmt_duration(secs):
    m, s = divmod(int(secs), 60)
    return f"{m}m{s:02d}s"


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


class Progress:
    """Counter with time-throttled ``progress: <name>`` lines.

    ``total`` may be a count of anything (batches, bytes); ``unit_scale``
    renders the counters as GiB instead of raw numbers.
    """

    def __init__(
        self, *, total, name, disable=False, min_interval=30.0, unit_scale=False
    ):
        self.total = total
        self.name = name
        self.disable = disable
        self.min_interval = min_interval
        self.unit_scale = unit_scale
        self.n = 0
        self.tic = time.time()
        self.last_print = self.tic

    def _fmt(self, n):
        if self.unit_scale:
            return f"{n / 2**30:.2f}GiB"
        return str(n)

    def update(self, k=1):
        self.n += k
        now = time.time()
        if now - self.last_print >= self.min_interval:
            self.last_print = now
            self._emit(now)

    def close(self):
        self._emit(time.time())

    def _emit(self, now):
        if self.disable:
            return
        pct = 100.0 * self.n / self.total if self.total else 100.0
        log(
            indent=1,
            progress=self.name,
            n=self._fmt(self.n),
            total=self._fmt(self.total),
            pct=f"{pct:.0f}",
            elapsed=fmt_duration(now - self.tic),
        )
