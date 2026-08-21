"""Run the smoke example, so a broken training path fails in CI rather than on
the cluster. Skipped without a GPU: the model's attention has no CPU backward.
"""

from datetime import datetime
from pathlib import Path

import pytest
import torch

from examples.smoke import smoke

PRE_DIR = "~/scratch/pre/relbench-preprocessed"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="training needs a GPU")
@pytest.mark.skipif(
    not Path(PRE_DIR).is_dir(), reason=f"no preprocessed data at {PRE_DIR}"
)
def test_smoke(tmp_path):
    run_id = f"{datetime.now():%y-%m-%d_%H-%M-%S}"
    smoke(
        pre_dir=PRE_DIR,
        out_root=str(tmp_path),
        run_id=run_id,
        total_steps=3,
        compile=False,
    )
    out = tmp_path / "no-entity" / "smoke" / run_id
    assert (out / "params.json").is_file(), "the run's arguments are its record"
    assert (out / "resume.pt").is_file(), (
        "a finished run must leave a resumable checkpoint"
    )
