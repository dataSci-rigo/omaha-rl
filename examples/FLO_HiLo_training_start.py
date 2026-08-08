import os
os.environ["OMP_NUM_THREADS"] = "1"

from PokerRL.game.games import FixedLimitOmahaHiLo
from PokerRL.game.wrappers import VanillaEnvBuilder

from DeepCFR.EvalAgentDeepCFR import EvalAgentDeepCFR
from DeepCFR.TrainingProfile import TrainingProfile
from DeepCFR.workers.driver.Driver import Driver

if __name__ == '__main__':
    # Heads-up, fixed-limit Omaha Hi/Lo (8-or-better). See the project plan
    # ("Training strategy & compute budget") for why: HU halves per-traversal
    # cost and is required by PokerRL's br/rlbr eval tooling (both assert
    # N_SEATS == 2). No agent_bet_set/lbr_args here -- fixed-limit games don't
    # use bet-size fractions, and this variant's rules class doesn't implement
    # the batched all-hands evaluator LBR needs (see OmahaHiLoRules), so
    # eval_methods is left empty; benchmark externally against the ABC/
    # Bayesian bots instead (see ~/Documents/omaha).
    ctrl = Driver(t_prof=TrainingProfile(name="FLO_HiLo_HU_dense_residual",
                                         nn_type="dense_residual",

                                         DISTRIBUTED=False,
                                         CLUSTER=False,
                                         n_learner_actor_workers=8,

                                         max_buffer_size_adv=1000000,
                                         export_each_net=False,
                                         checkpoint_freq=50,
                                         eval_agent_export_freq=10,

                                         n_actions_traverser_samples=3,
                                         n_traversals_per_iter=15000,
                                         n_batches_adv_training=750,
                                         max_n_las_sync_simultaneously=8,

                                         use_pre_layers_adv=True,
                                         n_cards_state_units_adv=192,
                                         n_merge_and_table_layer_units_adv=64,
                                         n_units_final_adv=64,
                                         lr_patience_adv=100,
                                         lr_adv=0.004,

                                         mini_batch_size_adv=2000,
                                         init_adv_model="last",  # last, random

                                         game_cls=FixedLimitOmahaHiLo,
                                         env_bldr_cls=VanillaEnvBuilder,
                                         n_seats=2,
                                         start_chips=2000,  # ~48x default fixed-limit stack of 48

                                         # SD-CFR (SINGLE) is already the default; leaving AVRG_NET
                                         # out keeps NN training cost roughly half of what it'd be
                                         # with both modes enabled.
                                         eval_modes_of_algo=(
                                             EvalAgentDeepCFR.EVAL_MODE_SINGLE,
                                         ),

                                         use_simplified_headsup_obs=True,

                                         log_verbose=True,
                                         ),
                  eval_methods={},
                  n_iterations=None)

    ctrl.run()
