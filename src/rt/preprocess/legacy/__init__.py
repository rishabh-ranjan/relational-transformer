"""Legacy (RT-v1-era) preprocessing: RelBench datasets with boolean columns
typed as a real Boolean semantic type, matching the released RT-v1 checkpoints'
training data. See :mod:`rt.preprocess.legacy.main`.
"""

from rt.preprocess.legacy.main import preprocess_one_legacy, transform_dataset

__all__ = ["preprocess_one_legacy", "transform_dataset"]
