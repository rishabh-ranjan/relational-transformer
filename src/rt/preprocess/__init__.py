"""Turn relbench-format datasets into the tensors RT trains on.

Entry points take their arguments directly -- there is no CLI; see
examples/preprocess.py for a script that calls them.
"""

from rt.preprocess.embed import TextEmbedder, embed_texts
from rt.preprocess.main import ls, many, one, preprocess_one, upload

__all__ = [
    "TextEmbedder",
    "embed_texts",
    "ls",
    "many",
    "one",
    "preprocess_one",
    "upload",
]
