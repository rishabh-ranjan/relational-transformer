"""Log-file friendly progress reporting: plain lines, no carriage returns.

Drop-in replacement for the handful of tqdm bars this repo used. Prints are
throttled in time so a long loop yields a bounded number of lines, and the
final line always lands so a log tail shows completion.
"""

import time


def fmt_duration(secs):
    m, s = divmod(int(secs), 60)
    return f"{m}m{s:02d}s"


class Progress:
    """Counter with time-throttled progress lines.

    ``total`` may be a count of anything (batches, bytes); ``unit_scale``
    renders the counters as GiB instead of raw numbers.
    """

    def __init__(
        self, *, total, desc, disable=False, min_interval=30.0, unit_scale=False
    ):
        self.total = total
        self.desc = desc
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
        print(
            f"{self.desc}: {self._fmt(self.n)}/{self._fmt(self.total)}"
            f" ({pct:.0f}%) in {fmt_duration(now - self.tic)}",
            flush=True,
        )
