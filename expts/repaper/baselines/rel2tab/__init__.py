"""rel2tab: label-matched tabular baselines over RT's eval contexts.

A baseline is a (featurizer, predictor) pair wrapped in
:class:`~expts.repaper.baselines.rel2tab.model.Rel2TabModel`, which plugs into
``rt.eval.evaluator.Evaluator`` exactly like an RT net: for every eval row it
extracts the labeled task rows visible in the row's sampled context, features
them, and predicts the target from them. The baselines therefore consume the
same contexts (and so the same labels) as the RT curve they are compared with.

Features are precomputed to disk once per (db, table) by the featurize scripts
beside this package (see ../README.md) and looked up by node index at eval
time via :class:`~expts.repaper.baselines.rel2tab.precomputed.PrecomputedFeaturizer`.
"""
