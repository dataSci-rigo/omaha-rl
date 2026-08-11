"""
Laptop-scaled Deep CFR / SD-CFR training for heads-up Pot Limit Omaha.

See ../TRAINING_PLAN.md for the sizing rationale. Launched nightly by
laptop/run_night.sh and auto-resumes from the latest checkpoint under PATH_DATA.

Set OMAHA_SMOKE=1 for a minutes-long end-to-end validation run with tiny buffers.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
os.environ["OMP_NUM_THREADS"] = "5"  # single-process run: let torch use most cores

import torch

from PokerRL.game.games import PLO
from PokerRL.eval.lbr.LBRArgs import LBRArgs
from PokerRL.game import bet_sets
from PokerRL.game.Poker import Poker
from PokerRL.game.wrappers import VanillaEnvBuilder

from DeepCFR.EvalAgentDeepCFR import EvalAgentDeepCFR
from DeepCFR.TrainingProfile import TrainingProfile
from DeepCFR.workers.driver.Driver import Driver

SMOKE = os.environ.get("OMAHA_SMOKE") == "1"

NAME = "PLO_smoke" if SMOKE else "PLO_laptop_v1"
PATH_DATA = os.path.expanduser(os.environ.get("OMAHA_PATH_DATA", "~/omaha_rl_data"))

N_ITERATIONS = 2 if SMOKE else 64
MAX_BUFFER_SIZE_ADV = 10_000 if SMOKE else 1_000_000
N_TRAVERSALS_PER_ITER = 50 if SMOKE else 50_000
N_BATCHES_ADV_TRAINING = 20 if SMOKE else 1000
MINI_BATCH_SIZE_ADV = 250 if SMOKE else 2500


def _find_latest_checkpoint():
    ckpt_root = os.path.join(PATH_DATA, "checkpoint")
    if not os.path.isdir(ckpt_root):
        return None, None
    best_step, best_name = None, None
    for run_name in os.listdir(ckpt_root):  # stored name may differ by trailing "_"s
        if run_name.rstrip("_") != NAME:
            continue
        run_dir = os.path.join(ckpt_root, run_name)
        for s in os.listdir(run_dir):
            # a complete 2-seat local checkpoint has 6 files (Chief/LA/PS x P0/P1);
            # a kill mid-save leaves fewer, and loading those would crash
            if s.isdigit() and len(os.listdir(os.path.join(run_dir, s))) >= 6:
                if best_step is None or int(s) > best_step:
                    best_step, best_name = int(s), run_name
    return best_step, best_name


if __name__ == '__main__':
    torch.set_num_threads(5)

    step_to_import, name_to_import = _find_latest_checkpoint()
    if step_to_import is not None:
        print("Resuming '%s' from checkpoint at iteration %d" % (name_to_import, step_to_import))
        if step_to_import >= N_ITERATIONS:
            print("Run already finished (%d/%d iterations). Nothing to do." % (step_to_import, N_ITERATIONS))
            raise SystemExit(0)

    ctrl = Driver(iteration_to_import=step_to_import,
                  name_to_import=name_to_import,
                  t_prof=TrainingProfile(name=NAME,
                                         path_data=PATH_DATA,
                                         nn_type="dense_residual",

                                         DISTRIBUTED=False,
                                         CLUSTER=False,
                                         n_learner_actor_workers=1,

                                         max_buffer_size_adv=MAX_BUFFER_SIZE_ADV,
                                         export_each_net=False,
                                         checkpoint_freq=2,  # Driver auto-deletes older checkpoints
                                         eval_agent_export_freq=4,

                                         n_actions_traverser_samples=4,
                                         n_traversals_per_iter=N_TRAVERSALS_PER_ITER,
                                         n_batches_adv_training=N_BATCHES_ADV_TRAINING,
                                         max_n_las_sync_simultaneously=20,

                                         use_pre_layers_adv=True,
                                         n_cards_state_units_adv=192,
                                         n_merge_and_table_layer_units_adv=64,
                                         n_units_final_adv=64,
                                         lr_patience_adv=350,
                                         lr_adv=0.004,

                                         mini_batch_size_adv=MINI_BATCH_SIZE_ADV,
                                         init_adv_model="last",

                                         game_cls=PLO,
                                         env_bldr_cls=VanillaEnvBuilder,
                                         agent_bet_set=bet_sets.PL_2,
                                         n_seats=2,
                                         start_chips=10000,

                                         eval_modes_of_algo=(
                                             EvalAgentDeepCFR.EVAL_MODE_SINGLE,  # SD-CFR
                                         ),

                                         use_simplified_headsup_obs=True,

                                         log_verbose=True,
                                         lbr_args=LBRArgs(lbr_bet_set=bet_sets.PL_2,
                                                          n_lbr_hands_per_seat=1,
                                                          lbr_check_to_round=Poker.TURN,
                                                          n_parallel_lbr_workers=1,
                                                          use_gpu_for_batch_eval=False,
                                                          DISTRIBUTED=False,
                                                          ),
                                         ),
                  eval_methods={
                      "": 99,  # as in the stock script: no lbr/br/h2h during training
                  },
                  n_iterations=N_ITERATIONS)

    ctrl.run()
