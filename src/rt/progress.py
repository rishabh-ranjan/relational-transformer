"""Log-file friendly progress reporting: plain lines, no carriage returns.

Every line this module emits -- and every line the callers below emit through
``log`` -- is a flat sequence of ``key: value`` pairs separated by two spaces,
the first pair always ``event: <name>``::

    event: progress  name: eval@1000  n: 12  total: 57  pct: 21  elapsed: 0m30s

Values never contain whitespace, so after stripping ANSI codes
(``re.sub(r"\\x1b\\[[0-9;]*m", "", line)``) a line parses into a dict with
``re.findall(r"(\\S+): (\\S+)", line)``. Values are bolded and column-padded,
and leading indentation nests records for human eyes; both are cosmetic.
"""

import time

BOLD = "\033[1m"
RESET = "\033[0m"


def fmt_duration(secs):
    m, s = divmod(int(secs), 60)
    return f"{m}m{s:02d}s"


_widths: dict[str, dict[str, int]] = {}


def log(event, *, indent=0, **fields):
    """Emit one ``key: value`` record. Values must be whitespace-free.

    Values are bolded, and padded to the widest value seen so far for that
    ``(event, key)``, so repeated records of one event line up in columns.
    """
    widths = _widths.setdefault(event, {})
    parts = [f"event: {BOLD}{event}{RESET}"]
    for k, v in fields.items():
        v = str(v)
        widths[k] = max(widths.get(k, 0), len(v))
        parts.append(f"{k}: {BOLD}{v.ljust(widths[k])}{RESET}")
    print(("  " * indent + "  ".join(parts)).rstrip(), flush=True)


class Progress:
    """Counter with time-throttled ``event: progress`` lines.

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
            "progress",
            indent=1,
            name=self.name,
            n=self._fmt(self.n),
            total=self._fmt(self.total),
            pct=f"{pct:.0f}",
            elapsed=fmt_duration(now - self.tic),
        )
