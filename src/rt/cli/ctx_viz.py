"""CLI for rt.ctx_viz. All defaults live here; see rt.ctx_viz for logic."""

import tyro

from rt.ctx_viz import Config, main


def default_config() -> Config:
    return Config(
        host="0.0.0.0",
        port=8765,
        pre_root="pre",
        quiet=False,
        port_fallback=True,
    )


if __name__ == "__main__":
    main(tyro.cli(Config, default=default_config()))
