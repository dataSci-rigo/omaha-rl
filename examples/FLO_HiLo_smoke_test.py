"""
Tiny, fast smoke test of the DeepCFR training pipeline (ray/torch workers,
traversal, checkpointing) against FixedLimitOmahaHiLo -- NOT a real training
run. Separate profile name from FLO_HiLo_training_start.py / the nightly
run so it doesn't touch their checkpoints.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"

from PokerRL.game.games import FixedLimitOmahaHiLo
from PokerRL.game.wrappers import VanillaEnvBuilder

from DeepCFR.EvalAgentDeepCFR import EvalAgentDeepCFR
from DeepCFR.TrainingProfile import TrainingProfile
from DeepCFR.workers.driver.Driver import Driver

if __name__ == '__main__':
    ctrl = Driver(t_prof=TrainingProfile(name="FLO_HiLo_smoke_test",
                                         nn_type="dense_residual",

                                         DISTRIBUTED=False,
                                         CLUSTER=False,
                                         n_learner_actor_workers=2,

                                         max_buffer_size_adv=10000,
                                         export_each_net=False,
                                         checkpoint_freq=1,
                                         eval_agent_export_freq=1,

                                         n_actions_traverser_samples=2,
                                         n_traversals_per_iter=50,
                                         n_batches_adv_training=5,
                                         max_n_las_sync_simultaneously=2,

                                         use_pre_layers_adv=True,
                                         n_cards_state_units_adv=192,
                                         n_merge_and_table_layer_units_adv=64,
                                         n_units_final_adv=64,
                                         lr_patience_adv=100,
                                         lr_adv=0.004,

                                         mini_batch_size_adv=64,
                                         init_adv_model="random",

                                         game_cls=FixedLimitOmahaHiLo,
                                         env_bldr_cls=VanillaEnvBuilder,
                                         n_seats=2,
                                         start_chips=2000,

                                         eval_modes_of_algo=(
                                             EvalAgentDeepCFR.EVAL_MODE_SINGLE,
                                         ),

                                         use_simplified_headsup_obs=True,

                                         log_verbose=True,
                                         ),
                  eval_methods={},
                  n_iterations=2)

    ctrl.run()
    print("\nSMOKE TEST PASSED: 2 iterations completed without error.\n")
