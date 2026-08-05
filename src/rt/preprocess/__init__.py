"""Turn relbench-format datasets into the tensors RT trains on.

Entry points take their arguments directly -- there is no CLI; see
examples/preprocess.py for a script that calls them.
"""

from rt.preprocess.embed import TextEmbedder, embed_texts
from rt.preprocess._preprocess import (
    dataset_name,
    embed_dataset,
    ls,
    many,
    one,
    preprocess_one,
    run_rustler_pre,
    update_meta_with_embeddings,
    upload,
)

# The pipeline's steps are exported alongside the whole-dataset entry points:
# `one` and `many` are one arrangement of them, not the only one. A caller that
# reads its raw data from somewhere other than where the data came from, or that
# wants slurm rather than `many`'s loop to schedule the work, composes the same
# steps -- and should not have to import them out of a private module to do it.
__all__ = [
    "TextEmbedder",
    "dataset_name",
    "embed_dataset",
    "embed_texts",
    "ls",
    "many",
    "one",
    "preprocess_one",
    "run_rustler_pre",
    "update_meta_with_embeddings",
    "upload",
]
