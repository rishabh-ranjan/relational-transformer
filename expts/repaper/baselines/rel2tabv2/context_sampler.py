import numpy as np


class ContextSampler:
    def sample(self, candidates, n_rows):
        raise NotImplementedError


class UniformRandomSampler(ContextSampler):
    def __init__(self, seed):
        self.seed = seed

    def sample(self, candidates, n_rows):
        candidates = np.asarray(candidates)
        assert n_rows <= len(candidates), (
            f"asked for {n_rows} context rows but only {len(candidates)} "
            f"candidates are available"
        )
        rng = np.random.default_rng(self.seed)
        return np.sort(rng.choice(candidates, size=n_rows, replace=False))


class MostRecentSampler(ContextSampler):
    def __init__(self, timestamps):
        self.timestamps = timestamps

    def sample(self, candidates, n_rows):
        candidates = np.asarray(candidates)
        assert n_rows <= len(candidates), (
            f"asked for {n_rows} context rows but only {len(candidates)} "
            f"candidates are available"
        )
        ts = np.asarray(self.timestamps)
        assert len(ts) == len(candidates), (
            f"{len(ts)} timestamps for {len(candidates)} candidates"
        )
        newest = np.argsort(ts, kind="stable")[::-1][:n_rows]
        return np.sort(candidates[newest])
