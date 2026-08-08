// Thin C ABI shim exposing nit's OmahaEightHandEvaluator to the ctypes
// convention used by PokerRL/game/_/cpp_wrappers/CppHandeval.py.
//
// Card encoding matches PokerRL/game/_/rl_env/game_rules_plo.py exactly:
//   rank: 0='2', 1='3', ..., 8='T', 9='J', 10='Q', 11='K', 12='A'
//   suit: 0..3 (arbitrary but consistent per PokerRL.SUIT_DICT)
//   Poker.CARD_NOT_DEALT_TOKEN_1D == -127 marks an undealt board card.
//
// nit::Rank(uint8_t) and nit::Suit(uint8_t) both accept a raw index in
// [0, N-1] directly (see nit/eval/rank.cc, nit/eval/suit.cc) rather than
// requiring ASCII chars, so PokerRL's rank/suit ints can be passed straight
// through with no translation. Suit *labels* differ between the two
// projects (PokerRL: 0=h,1=d,2=s,3=c; nit: 0=c,1=d,2=h,3=s) but that's
// immaterial for hand evaluation -- only "same suit vs. different suit"
// matters for flush detection, and passing PokerRL's raw index consistently
// for every card preserves that.

#include <cstdint>

#include "nit/eval/card.h"
#include "nit/eval/card_set.h"
#include "nit/eval/omaha_eight_hand_evaluator.h"

namespace {

constexpr int8_t CARD_NOT_DEALT = -127;
constexpr int N_HOLE_CARDS = 4;
constexpr int N_BOARD_CARDS = 5;

nit::Card make_card(const int8_t* rank_suit) {
    return nit::Card(nit::Rank(static_cast<uint8_t>(rank_suit[0])),
                     nit::Suit(static_cast<uint8_t>(rank_suit[1])));
}

}  // namespace

extern "C" {

// hand_2d:  4 rows of [rank, suit]      -- the 4 Omaha hole cards, always dealt.
// board_2d: 5 rows of [rank, suit]      -- board cards; undealt slots are
//                                          marked with CARD_NOT_DEALT and skipped.
// hi_rank_out: order-preserving int, higher is better (nit::PokerEvaluation::code()).
// lo_rank_out: same convention, or -1 if no qualifying (8-or-better) low exists.
void get_hand_rank_52_plo8(const int8_t* const* hand_2d,
                           const int8_t* const* board_2d,
                           int32_t* hi_rank_out,
                           int32_t* lo_rank_out) {
    nit::CardSet hand;
    for (int i = 0; i < N_HOLE_CARDS; ++i) {
        hand.insert(make_card(hand_2d[i]));
    }

    nit::CardSet board;
    for (int i = 0; i < N_BOARD_CARDS; ++i) {
        if (board_2d[i][0] == CARD_NOT_DEALT) {
            continue;
        }
        board.insert(make_card(board_2d[i]));
    }

    nit::OmahaEightHandEvaluator evaluator;
    nit::PokerHandEvaluation result = evaluator.evaluateHand(hand, board);

    *hi_rank_out = result.high().code();
    *lo_rank_out = result.highlow() ? result.low().code() : -1;
}

}  // extern "C"
