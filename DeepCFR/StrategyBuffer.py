# Copyright (c) 2019 Eric Steinberger


import numpy as np

from DeepCFR.IterationStrategy import IterationStrategy


class StrategyBuffer:

    def __init__(self, t_prof, owner, env_bldr, device, max_size=None):
        self._t_prof = t_prof
        self._env_bldr = env_bldr
        self._owner = owner
        self._device = device

        self._max_size = max_size

        self._strategies = []
        self._weights = []
        self._size = 0
        self._last_cfr_iter_seen = None

    @property
    def owner(self):
        return self._owner

    @property
    def size(self):
        return self._size

    @property
    def device(self):
        return self._device

    @property
    def strategies(self):
        return self._strategies

    @property
    def last_cfr_iter_seen(self):
        return self._last_cfr_iter_seen

    @property
    def max_size(self):
        return self._max_size

    def get(self, i):
        return self._strategies[i]

    def get_strats_and_weights(self):
        return zip(self._strategies, self._weights)

    def sample_strat_weighted(self):
        return self.get(self.sample_strat_idx_weighted())

    def sample_strat_idx_weighted(self):
        if self._size == 0:
            return None

        w = np.array(self._weights)
        s = np.sum(w)
        w = np.full_like(w, fill_value=1 / w.shape[0]) if s == 0 else w / s

        return np.random.choice(a=np.arange(start=0, stop=self._size, dtype=np.int32),
                                p=w)

    def add(self, iteration_strat):
        if self._max_size is None or (self._size < self._max_size):
            self._strategies.append(iteration_strat.get_copy(device=self._device))
            self._weights.append(iteration_strat.cfr_iteration + 1)

            self._size = len(self._strategies)

        elif np.random.random() < (float(self._max_size) / float(self._last_cfr_iter_seen)):
            idx = np.random.randint(len(self._strategies))
            self._strategies[idx] = iteration_strat.get_copy(device=self._device)
            self._weights[idx] = iteration_strat.cfr_iteration + 1

        self._last_cfr_iter_seen = iteration_strat.cfr_iteration

    def state_dict(self):
        return {
            "nets": [(s.net_state_dict(), s.cfr_iteration) for s in self._strategies],
            "owner": self.owner,
            # add() divides by this once the buffer is full. Without persisting it a
            # resumed bounded buffer starts at None and raises TypeError on the first
            # overflow. Same fix as _ReservoirBufferBase's "last_cfr_iteration_seen".
            "last_cfr_iter_seen": self._last_cfr_iter_seen,
        }

    def load_state_dict(self, state):
        assert self.owner == state["owner"]

        saved_nets = state["nets"]
        if self._max_size is not None and len(saved_nets) > self._max_size:
            # A checkpoint can hold more nets than the current bound (e.g. it was saved
            # unbounded, or with a larger max_size). Appending them all would leave the
            # buffer permanently over capacity, since add() only ever replaces in place.
            # Subsample uniformly WITHOUT replacement, keeping each net's stored
            # cfr_iteration, which is the same estimator add()'s reservoir maintains.
            # Sample before constructing IterationStrategy objects -- each one builds a
            # live network, so trimming afterwards would not lower peak memory.
            keep = np.random.choice(len(saved_nets), size=self._max_size, replace=False)
            saved_nets = [saved_nets[i] for i in sorted(keep)]

        self._strategies = []
        self._weights = []  # was not reset -- a second load appended onto stale weights
        for net_state_dict, cfr_iter in saved_nets:
            s = IterationStrategy(t_prof=self._t_prof, owner=self.owner, env_bldr=self._env_bldr,
                                  device=self._device, cfr_iter=cfr_iter)
            s.load_net_state_dict(net_state_dict)
            self._strategies.append(s)
            # Must match add()'s `cfr_iteration + 1`. Storing the bare cfr_iter here
            # meant every checkpoint resume silently reweighted the whole buffer, and
            # gave the iteration-0 net weight 0 so sample_strat_idx_weighted() could
            # never select it. Since the nightly run resumes every cycle, every agent
            # exported since has carried these shifted weights.
            self._weights.append(cfr_iter + 1)

        self._size = len(self._strategies)

        # Restore the reservoir's acceptance denominator. .get() keeps older
        # checkpoints (saved before this key existed) loadable; for those, fall back
        # to the highest saved iteration, which is what it would have been. Leaving
        # it None makes add() raise TypeError on float(None) the first time a full
        # buffer overflows -- which only happens once max_size is set, hence this
        # became mandatory when the Chief's buffer was bounded.
        self._last_cfr_iter_seen = state.get("last_cfr_iter_seen", None)
        if self._last_cfr_iter_seen is None and state["nets"]:
            self._last_cfr_iter_seen = max(cfr_iter for _, cfr_iter in state["nets"])  # full set, not the subsample
