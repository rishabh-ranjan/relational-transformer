"""Evaluation: shared Evaluator kernel, metrics, relbench scoring, drivers."""

from rt.eval.evaluator import Evaluator
from rt.eval._eval import (
    build_evaluator,
    main,
    member_context_seed,
    run_and_report,
    run_ensemble,
)
from rt.eval.metrics import metric_for

__all__ = [
    "member_context_seed",
    "Evaluator",
    "build_evaluator",
    "main",
    "metric_for",
    "run_and_report",
    "run_ensemble",
]
