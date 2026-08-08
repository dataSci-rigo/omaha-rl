"""
Play Fixed-Limit Omaha Hi/Lo (8-or-better) against yourself/a friend in a
terminal -- for verifying the fixed small-bet/big-bet sizing, raise cap, and
hi/lo split-pot behavior by hand.
"""

from PokerRL.game.InteractiveGame import InteractiveGame
from PokerRL.game.games import FixedLimitOmahaHiLo

if __name__ == '__main__':
    game_cls = FixedLimitOmahaHiLo
    args = game_cls.ARGS_CLS(n_seats=2,
                             starting_stack_sizes_list=[48 * game_cls.BIG_BET] * 2,
                             stack_randomization_range=(0, 0))

    game = InteractiveGame(env_cls=game_cls,
                           env_args=args,
                           seats_human_plays_list=[0, 1])
    game.start_to_play()
