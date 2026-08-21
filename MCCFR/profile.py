"""
Minimal picklable training-profile stand-in for the MCCFR pipeline.

EvalAgentBase and rl_util.get_env_builder only touch a handful of t_prof
fields; carrying a full DeepCFR TrainingProfile (torch modules, ray flags,
buffer configs) into a tabular agent's pickle would be pure baggage. This
class has exactly the fields they read.
"""


class MCCFRProfile:

    def __init__(self, name, env_args=None, k_postflop=50, n_rollouts=32,
                 bucketer_kind="m1", n_seats=2):
        self.name = name
        self.DISTRIBUTED = False
        self.CLUSTER = False
        self.game_cls_str = "FixedLimitOmahaHiLo"
        self.env_builder_cls_str = "VanillaEnvBuilder"
        self.device_inference = "cpu"
        self.n_seats = n_seats
        if env_args is None:
            from PokerRL.game.games import FixedLimitOmahaHiLo
            env_args = FixedLimitOmahaHiLo.ARGS_CLS(
                n_seats=n_seats,
                starting_stack_sizes_list=[2000] * n_seats,
                stack_randomization_range=(0, 0))
        self.module_args = {"env": env_args}
        # bucketer config, so a loaded eval agent can rebuild its abstraction
        self.bucketer_kind = bucketer_kind
        self.k_postflop = k_postflop
        self.n_rollouts = n_rollouts
