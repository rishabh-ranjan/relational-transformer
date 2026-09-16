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
        # Random order, deliberately unsorted: the caller nests sizes by taking
        # prefixes, and a prefix of a random permutation is itself a uniform
        # subset. Sorting here made drawn[:64] the 64 lowest node indices, which
        # on rel-f1 is 1950-1954 rather than a sample of 1950-2004.
        return rng.choice(candidates, size=n_rows, replace=False)


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


class BernoulliSampler(ContextSampler):
    def __init__(self, seed, probability):
        assert 0.0 < probability <= 1.0, f"probability {probability} not in (0, 1]"
        self.seed = seed
        self.probability = probability

    def sample(self, candidates, n_rows):
        # n_rows is ignored: the size is what the coin flips give,
        # |context| ~ Binomial(len(candidates), probability). One uniform per
        # candidate, thresholded, so a family of probabilities drawn from the
        # same seed is nested -- every row in the p=0.01 context is in the
        # p=0.1 one.
        candidates = np.asarray(candidates)
        rng = np.random.default_rng(self.seed)
        keep = rng.random(len(candidates)) < self.probability
        return candidates[keep]
