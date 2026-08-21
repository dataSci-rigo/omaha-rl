# Copyright (c) 2019 Eric Steinberger


import numpy as np
import torch


class ReservoirBufferBase:

    def __init__(self, owner, max_size, env_bldr, nn_type, iter_weighting_exponent):
        self._owner = owner
        self._env_bldr = env_bldr
        self.device = torch.device("cpu")

        self._owner = owner
        self._env_bldr = env_bldr
        self._max_size = max_size
        self._nn_type = nn_type
        self.device = torch.device("cpu")
        self.size = 0
        self.n_entries_seen = 0

        if nn_type == "recurrent":
            self._pub_obs_buffer = np.empty(shape=(max_size,), dtype=object)
        elif nn_type == "feedforward":
            self._pub_obs_buffer = torch.zeros((max_size, self._env_bldr.pub_obs_size), dtype=torch.float32,
                                               device=self.device)
        elif nn_type == "convolutional":
            self._pub_obs_buffer = torch.zeros((max_size, self._env_bldr.pub_obs_size), dtype=torch.float32,
                                               device=self.device)
        elif nn_type == "dense_residual":
            self._pub_obs_buffer = torch.zeros((max_size, self._env_bldr.pub_obs_size), dtype=torch.float32,
                                               device=self.device)
        else:
            raise ValueError(nn_type)

        self._range_idx_buffer = torch.zeros((max_size,), dtype=torch.long, device=self.device)
        self._legal_action_mask_buffer = torch.zeros((max_size, env_bldr.N_ACTIONS,),
                                                     dtype=torch.float32, device=self.device)
        self._iteration_buffer = torch.zeros((max_size,), dtype=torch.float32, device=self.device)
        self._iter_weighting_exponent = iter_weighting_exponent

        self._last_cfr_iteration_seen = None

    def add(self, **kwargs):
        """
        Dont forget to n_entries_seen+=1 !!
        """
        raise NotImplementedError

    def sample(self, batch_size, device):
        raise NotImplementedError

    def _should_add(self):
        return np.random.random() < (float(self._max_size) / float(self.n_entries_seen))

    def _np_to_torch(self, arr):
        return torch.from_numpy(np.copy(arr)).to(self.device)

    def _random_idx(self):
        return np.random.randint(low=0, high=self._max_size)

    def state_dict(self):
        return {
            "owner": self._owner,
            "max_size": self._max_size,
            "nn_type": self._nn_type,
            "size": self.size,
            "n_entries_seen": self.n_entries_seen,
            "iter_weighting_exponent": self._iter_weighting_exponent,
            # sample() divides by this; without persisting it, a resumed
            # buffer starts at None and crashes the first time sample() is
            # called before this player's buffer sees a fresh add() in the
            # new process.
            "last_cfr_iteration_seen": self._last_cfr_iteration_seen,

            "pub_obs_buffer": self._pub_obs_buffer,
            "range_idx_buffer": self._range_idx_buffer,
            "legal_action_mask_buffer": self._legal_action_mask_buffer,
            "iteration_buffer": self._iteration_buffer,
        }

    def load_state_dict(self, state):
        assert self._owner == state["owner"]
        if not self._nn_type == state["nn_type"]:
            print('Current AdvNet type differs from that one of checkpoint!')

        saved_max_size = state["max_size"]
        # Growing max_buffer_size_adv used to be impossible: this asserted equality,
        # so any change orphaned every existing checkpoint and forced training to
        # restart from iteration 0. That mattered because the buffer had been shrunk
        # to 75,000 to survive a memory problem that turned out to be per-net LUT
        # duplication (see PokerRL/rl/neural/_shared_luts.py), leaving each iteration
        # drawing 1.5M training samples from a 75k buffer -- a 20:1 resample ratio.
        # Growing is safe (the saved entries fit); shrinking is not, since it would
        # need a principled subsample, so it stays refused.
        assert saved_max_size <= self._max_size, (
            f"cannot load a buffer of max_size {saved_max_size} into one of "
            f"{self._max_size}: shrinking would require subsampling the saved entries")
        migrating = saved_max_size < self._max_size

        self.size = state["size"]
        self.n_entries_seen = state["n_entries_seen"]
        self._iter_weighting_exponent = state["iter_weighting_exponent"]
        # .get() for backward compat with checkpoints saved before this key existed
        self._last_cfr_iteration_seen = state.get("last_cfr_iteration_seen", None)

        on_device = self._nn_type != "recurrent"  # recurrent keeps a numpy object array

        def _restore(attr, key):
            saved = state[key]
            if not migrating:
                setattr(self, attr, saved.to(self.device) if on_device else saved)
                return
            # Copy the saved entries into the front of the already-allocated larger
            # buffer. add() then appends into the free tail until it fills, so the
            # enlarged buffer ingests fresh data at 100% acceptance during that
            # phase -- the same fill behaviour a new buffer has, and a deliberate
            # side benefit here: n_entries_seen is preserved, so once full it
            # returns to correct reservoir acceptance (max_size / n_entries_seen).
            buf = getattr(self, attr)
            buf[:self.size] = saved[:self.size].to(self.device) if on_device else saved[:self.size]

        _restore("_pub_obs_buffer", "pub_obs_buffer")
        _restore("_range_idx_buffer", "range_idx_buffer")
        _restore("_legal_action_mask_buffer", "legal_action_mask_buffer")
        _restore("_iteration_buffer", "iteration_buffer")

        if migrating:
            print(f"Migrated reservoir buffer P{self._owner}: max_size {saved_max_size} -> "
                  f"{self._max_size}, kept {self.size} entries, n_entries_seen={self.n_entries_seen}")
