# Copyright (c) 2019 Eric Steinberger 2020 Vsevolod Kompantsev

"""
finished, tested, works properly for HU game series with hand history logging.
Changeable arguments are
-- players names *Hero and Dummy by default*
-- strings for HH file headers, which set game type, blinds, table name
and so on in order to make HH file properly readable by PT4.
"""

import numpy as np
from PokerRL.game.hh_log import HandHistoryLogger


class AgentTournament:

    def __init__(self, env_cls, env_args, eval_agent_1, eval_agent_2, logfile=None):
        self._eval_agents = [eval_agent_1, eval_agent_2]

        self._env_cls = env_cls
        self._env_args = env_args
        self._lut_holder = self._env_cls.get_lut_holder()

        # here we determine do we have to log our games into .txt or not
        # game variants: game_type="Omaha Pot Limit ($0.5/$1 USD)"
        # or game_type="Hold'em No Limit ($0.5/$1 USD)"
        if logfile is not None:
            self._logger = HandHistoryLogger(logfile=logfile, game_type="Omaha Pot Limit ($0.5/$1 USD)",
                                             tablename_type="Table 'Chort IX' 6-max",
                                             divisor=env_cls.EV_NORMALIZER * 10,
                                             output_format="stars")
        else:
            self._logger = None
        assert env_args.n_seats == 2

    def run(self, n_games_per_seat):
        """
        Mirrored-deck ("duplicate poker") evaluation: each deck is dealt once and
        played TWICE with the agents' seats swapped, and one sample = the mean of
        the two legs. Card luck largely cancels within each pair, so the 95% CI
        shrinks substantially at the same hand count versus independent deals.
        (The agents are frozen at eval time -- forward passes only -- so replaying
        a deck cannot teach them anything.)

        n_games_per_seat = number of deck PAIRS; total hands = 2x that, same as
        the previous independent-deal scheme. Return contract unchanged:
        (mean, upper95, lower95) -- note the order.
        """
        REFERENCE_AGENT = 0

        _env = self._env_cls(env_args=self._env_args, is_evaluating=True,
                             lut_holder=self._lut_holder, hh_logger=self._logger)
        pair_means = np.empty(shape=(n_games_per_seat,), dtype=np.float32)

        for _hand_nr in range(n_games_per_seat):
            deck_state_dict = None

            leg_rews = []
            for seat_p0 in range(_env.N_SEATS):
                seat_p1 = 1 - seat_p0

                # names follow the rotation so hand histories stay readable
                if self._logger is not None:
                    self._logger.set_names(('Hero', 'Dummy') if seat_p0 == REFERENCE_AGENT
                                           else ('Dummy', 'Hero'))

                # """""""""""""""""
                # Reset -- leg 1 deals fresh, leg 2 replays the same deck
                # """""""""""""""""
                _, r_for_all, done, info = _env.reset(deck_state_dict=deck_state_dict)
                if deck_state_dict is None:
                    deck_state_dict = _env.cards_state_dict()

                for e in self._eval_agents:
                    e.reset(deck_state_dict=deck_state_dict)

                # """""""""""""""""
                # Play Episode
                # """""""""""""""""
                while not done:
                    p_id_acting = _env.current_player.seat_id

                    if p_id_acting == seat_p0:
                        action_int, _ = self._eval_agents[REFERENCE_AGENT].get_action(step_env=True, need_probs=False)
                        self._eval_agents[1 - REFERENCE_AGENT].notify_of_action(p_id_acted=p_id_acting,
                                                                                action_he_did=action_int)

                    elif p_id_acting == seat_p1:
                        action_int, _ = self._eval_agents[1 - REFERENCE_AGENT].get_action(step_env=True,
                                                                                          need_probs=False)
                        self._eval_agents[REFERENCE_AGENT].notify_of_action(p_id_acted=p_id_acting,
                                                                            action_he_did=action_int)

                    else:
                        raise ValueError("Only HU is supported!")

                    _, r_for_all, done, info = _env.step(action_int)

                leg_rews.append(r_for_all[seat_p0] * _env.REWARD_SCALAR * _env.EV_NORMALIZER)

            pair_means[_hand_nr] = 0.5 * (leg_rews[0] + leg_rews[1])
            if _hand_nr % 100 == 0:
                print(f"Hand: {_hand_nr} out of {n_games_per_seat}")

        mean = np.mean(pair_means).item()
        std = np.std(pair_means).item()

        # CI over mirrored PAIRS (each pair is one sample)
        _d = 1.96 * std / np.sqrt(n_games_per_seat)
        lower_conf95 = mean - _d
        upper_conf95 = mean + _d

        print(f"Played {n_games_per_seat * 2} hands of poker (mirrored decks).")
        print("Player 1", self._eval_agents[REFERENCE_AGENT].get_mode() + ":", mean, "milliBB per hand +/-", _d)
        print("Player 2", self._eval_agents[1 - REFERENCE_AGENT].get_mode() + ":", (-mean), "milliBB per hand+/-", _d)

        return float(mean), float(upper_conf95), float(lower_conf95)
