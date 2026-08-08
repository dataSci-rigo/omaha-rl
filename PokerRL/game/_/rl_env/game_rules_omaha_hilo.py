from PokerRL.game._.rl_env.game_rules_plo import PLORules

"""
classes in this file are utilities that poker environments based on PokerEnv can inherit from. They override PokerEnv's
class variables and thereby set it to a certain rule set.
"""


class OmahaHiLoRules(PLORules):
    """
    Omaha Hi/Lo (8-or-better) -- same 4-hole-card / must-use-exactly-2 deal as PLO,
    but hand ranking produces a (hi_rank, lo_rank_or_none) pair instead of one scalar,
    and showdown splits each pot between the best hi hand and the best qualifying
    (8-or-better) lo hand (see PokerEnv.USES_HI_LO).
    """

    USES_HI_LO = True

    STRING = "OMAHA_HI_LO_RULES"

    def __init__(self):
        from PokerRL.game._.cpp_wrappers.CppHandEvalHiLo import CppHandEvalHiLo

        self._clib_hilo = CppHandEvalHiLo()

    def get_hand_rank(self, hand_2d, board_2d):
        """
        for docs refer to PokerEnv

        Returns:
            (int, int or None): (hi_rank, lo_rank). Higher is better for both. lo_rank
                is None if no 8-or-better low hand exists. Only comparable to ranks
                produced by this same evaluator -- do not mix with PLORules/CppHandeval
                hi ranks, which use an unrelated integer scale.
        """
        return self._clib_hilo.get_hand_rank_52_plo8(hand_2d=hand_2d, board_2d=board_2d)

    def get_hand_rank_all_hands_on_given_boards(self, boards_1d, lut_holder):
        """
        Not implemented for Omaha Hi/Lo -- nit's evaluator has no batch
        "every possible hole hand vs. this board" path like the compiled PLO hi
        evaluator does. Only needed by PokerRL/eval/lbr/LocalLBRWorker.py (Local
        Best Response evaluation); training and showdown don't call this.
        """
        raise NotImplementedError(
            "OmahaHiLoRules has no batched all-hands evaluator yet; LBR-style "
            "evaluation isn't supported for this variant."
        )

    @classmethod
    def get_lut_holder(cls):
        from PokerRL.game._.look_up_table import LutHolderPLO

        return LutHolderPLO(cls)
