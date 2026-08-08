# Copyright (c) 2026, added for Omaha Hi/Lo (8-or-better) support.

import ctypes
import os
from os.path import join as ospj

from PokerRL._.CppWrapper import CppWrapper


class CppHandEvalHiLo(CppWrapper):
    """
    Wraps lib_hand_eval_hilo.so, a thin shim over the vendored `nit`
    (github.com/rakhimov/nit) OmahaEightHandEvaluator. Unlike CppHandeval's
    get_hand_rank_52_plo (hi only), this evaluates hi and lo in one call --
    nit's evaluator handles the "use exactly 2 of 4 hole cards" combinatorics
    for both internally.
    """

    def __init__(self):
        super().__init__(path_to_dll=ospj(os.path.dirname(os.path.realpath(__file__)),
                                          "lib_hand_eval_hilo." + self.CPP_LIB_FILE_ENDING))
        self._clib.get_hand_rank_52_plo8.argtypes = [
            self.ARR_2D_ARG_TYPE,
            self.ARR_2D_ARG_TYPE,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
        ]
        self._clib.get_hand_rank_52_plo8.restype = None

    def get_hand_rank_52_plo8(self, hand_2d, board_2d):
        """
        Args:
            hand_2d (np.ndarray(shape=[4,2], dtype=int8)):  [rank, suit] x 4 hole cards.
            board_2d (np.ndarray(shape=[5,2], dtype=int8)): [rank, suit] x up to 5 board
                cards; undealt slots must use Poker.CARD_NOT_DEALT_TOKEN_2D.

        Returns:
            (int, int or None): (hi_rank, lo_rank). Higher is better for both, and they
                are only comparable to ranks produced by this same evaluator (their
                integer scale is unrelated to CppHandeval's PLO hi-only scale). lo_rank
                is None if no 8-or-better low hand exists.
        """
        hi = ctypes.c_int32()
        lo = ctypes.c_int32()
        self._clib.get_hand_rank_52_plo8(self.np_2d_arr_to_c(hand_2d), self.np_2d_arr_to_c(board_2d),
                                         ctypes.byref(hi), ctypes.byref(lo))
        return hi.value, (lo.value if lo.value != -1 else None)
