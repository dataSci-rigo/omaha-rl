"""
Inspect a tabular MCCFR checkpoint: the debuggability that motivated going
tabular. Prints per-street coverage and the average strategy at chosen spots.

    python3 examples/mccfr_inspect.py --profile MCCFR_M1            # summary
    python3 examples/mccfr_inspect.py --profile MCCFR_M1 --hand "As 2s Ah 3d"

The --hand view shows the preflop root strategy for that exact hand (via its
suit-isomorphism class) plus the response strategies facing a raise.
"""
import os

os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from MCCFR.abstraction.preflop_iso import (build_idx_to_hole_cards,  # noqa: E402
                                           build_hole_cards_to_idx,
                                           build_preflop_class_map,
                                           range_idx_of)
from MCCFR.betting_tree import build_tree, DECISION, CHECK_CALL, BET_RAISE  # noqa: E402

DATA_PATH = os.path.expanduser(os.environ.get("MCCFR_DATA_PATH",
                                              "~/poker_ai_data"))
STREETS = ["preflop", "flop", "turn", "river"]
RANKS, SUITS = "23456789TJQKA", "hdsc"


def parse_hand(spec):
    return [RANKS.index(c[0]) * 4 + SUITS.index(c[1]) for c in spec.split()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="MCCFR_M1")
    ap.add_argument("--hand", default=None, help='e.g. "As 2s Ah 3d"')
    args = ap.parse_args()

    path = os.path.join(DATA_PATH, "checkpoint", args.profile,
                        "mccfr_state.npz")
    tree = build_tree()
    with np.load(path) as data:
        import json
        meta = json.loads(bytes(data["meta"]).decode())
        avg = [data[f"avg_{s}"] for s in range(4)]

    K = meta["K"]
    print(f"profile {args.profile}: iteration {meta['iteration']}, K={K}")
    for s in range(4):
        sums = avg[s].sum(axis=1)
        n = len(sums)
        touched = int((sums > 0).sum())
        print(f"  {STREETS[s]:8s} {touched:>10,}/{n:<12,} infosets touched "
              f"({touched / n:.1%})")

    if args.hand:
        hole = parse_hand(args.hand)
        idx_to_hc = build_idx_to_hole_cards()
        hc_to_idx = build_hole_cards_to_idx(idx_to_hc)
        cls = build_preflop_class_map(idx_to_hc)[range_idx_of(hole, hc_to_idx)]
        print(f"\nhand [{args.hand}] -> preflop class {cls}")

        def show(node, label):
            row = avg[0][tree.decision_idx[node] * K[0] + cls]
            s = row.sum()
            if s <= 0:
                print(f"  {label:34s} (never visited)")
                return
            p = row / s
            print(f"  {label:34s} fold {p[0]:.3f}  call {p[1]:.3f}  "
                  f"raise {p[2]:.3f}")

        root = tree.ROOT
        show(root, "SB first action")
        vs_raise = int(tree.children[root, BET_RAISE])
        if tree.node_type[vs_raise] == DECISION:
            show(vs_raise, "BB facing SB raise")
        limp = int(tree.children[root, CHECK_CALL])
        if tree.node_type[limp] == DECISION:
            show(limp, "BB after SB limp")


if __name__ == "__main__":
    main()
