"""torchrun with the elastic agent deaf to SIGTERM.

Slurm signals every process in the job at preemption, so the agent gets SIGTERM
at the same instant as the ranks and starts killing them before they reach the
step boundary where they write resume.pt. Ignoring it in the *launcher* leaves
the ranks their own handlers and lets them save; the launcher is torn down
afterwards by the batch script, once resume.pt has landed.

Setting SIG_IGN is not enough on its own -- torch installs its handler after we
start -- so signal.signal is wrapped to refuse SIGTERM registrations. Only this
process is affected: the ranks are separate processes and keep their handlers.
"""

import signal
import sys

from torch.distributed.run import main

_real_signal = signal.signal


def _refuse_sigterm(sig, handler):
    if sig == signal.SIGTERM:
        return _real_signal(sig, signal.SIG_IGN)
    return _real_signal(sig, handler)


if __name__ == "__main__":
    _real_signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal = _refuse_sigterm
    main(sys.argv[1:])
