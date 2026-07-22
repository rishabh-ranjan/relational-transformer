"""Interactive web UI for inspecting rustler context/batch tensors.

Serves an HTTP UI (default port 8765) over preprocessed data (--pre-root,
local dir or Hub repo) for browsing sampled contexts token by token.
"""

import tyro

from rt.ctx_viz import Config, main


def default_config() -> Config:
    return Config(
        host="0.0.0.0",
        port=8765,
        pre_root="stanford-star/relbench-preprocessed",
        quiet=False,
        port_fallback=True,
    )


if __name__ == "__main__":
    main(tyro.cli(Config, default=default_config(), description=__doc__))
