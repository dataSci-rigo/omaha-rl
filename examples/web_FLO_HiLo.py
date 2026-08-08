"""
Fixed-Limit Omaha Hi/Lo (8-or-better) -- browser web server for this repo's
own FixedLimitOmahaHiLo env (not PokerKit). Two modes:
  - vs Bot:    play against ABCBot or BayesianBot (from ~/Documents/omaha)
               as a dry run for eventually playing the trained EvalAgent.
  - vs Friend: two humans, each on their own device (e.g. over Tailscale),
               each seeing only their own hole cards.

Run:  python3 examples/web_FLO_HiLo.py
Then open http://<this machine's tailscale ip>:5050/
"""
import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"

# Make this runnable from anywhere (not just the repo root with PYTHONPATH=.
# set, which is what the other examples/*.py scripts silently assume).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

sys.path.insert(0, os.path.expanduser("~/Documents/omaha"))
from abc_omaha_hilo import ABCBot  # noqa: E402
from bayesian_bot import BayesianBot  # noqa: E402
from pokerkit import Card as PKCard  # noqa: E402

from flask import Flask, jsonify, render_template_string, request  # noqa: E402

from PokerRL.game.games import FixedLimitOmahaHiLo  # noqa: E402
from PokerRL.game.Poker import Poker  # noqa: E402

STARTING_STACK = 48 * FixedLimitOmahaHiLo.BIG_BET
HUMAN_SEAT = 0
BOT_SEAT = 1

RANKS = "23456789TJQKA"
SUITS = "hdsc"


def card_str(c):
    return RANKS[c[0]] + SUITS[c[1]]


def cards_str(cards_2d):
    return "".join(card_str(c) for c in cards_2d if c[0] != Poker.CARD_NOT_DEALT_TOKEN_1D)


def make_bot(bot_type):
    if bot_type == "bayesian":
        return BayesianBot(name="Bayesian")
    return ABCBot()


class Game:
    def __init__(self):
        self.env = None
        self.done = True
        self.mode = None  # "bot" or "friend"
        self.bot = None
        self.bot_type = None
        self.history = []
        # PokerEnv's is_evaluating=True resets every player to a fixed
        # baseline stack on each .reset() (see _PokerPlayer.reset()) --
        # intentional for RL evaluation (clean per-hand outcomes), but not
        # what a human session wants by default. continue_stacks carries
        # each hand's net result forward into the next hand's starting
        # stack instead of resetting to baseline every time.
        self.continue_stacks = True
        self.session_over = False

    def new_game(self, mode, bot_type="abc", continue_stacks=True):
        args = FixedLimitOmahaHiLo.ARGS_CLS(n_seats=2,
                                            starting_stack_sizes_list=[STARTING_STACK] * 2,
                                            stack_randomization_range=(0, 0))
        self.env = FixedLimitOmahaHiLo(env_args=args, lut_holder=FixedLimitOmahaHiLo.get_lut_holder(),
                                       is_evaluating=True)
        self.env.reset()
        self.done = False
        self.mode = mode
        self.history = []
        self.bot = make_bot(bot_type) if mode == "bot" else None
        self.bot_type = bot_type if mode == "bot" else None
        self.continue_stacks = continue_stacks
        self.session_over = False

    def next_hand(self):
        if self.continue_stacks:
            deltas = [p.stack - STARTING_STACK for p in self.env.seats]
            # Bail out (without dealing) if carrying forward would leave
            # someone unable to post a big blind -- they need a stack reset.
            if any(STARTING_STACK + d <= FixedLimitOmahaHiLo.BIG_BET for d in deltas):
                self.session_over = True
                return
        self.env.reset()
        if self.continue_stacks:
            for p, d in zip(self.env.seats, deltas):
                p.stack += d
        self.done = False
        self.history = []

    def bot_decision(self, seat):
        env = self.env
        hole = list(PKCard.parse(cards_str(env.seats[seat].hand)))
        board = list(PKCard.parse(cards_str(env.board)))
        to_call = env._get_biggest_bet_out_there_aka_total_to_call() - env.seats[seat].current_bet
        # main_pot only holds bets already swept between rounds -- add bets
        # committed so far this round (e.g. blinds preflop) so the bot's
        # pot-odds math sees the real pot, not an artificially empty one.
        pot = env.main_pot + sum(p.current_bet for p in env.seats)
        can_raise = Poker.BET_RAISE in env.get_legal_actions()
        street = Poker.INT2STRING_ROUND[env.current_round]
        decision = self.bot.decide(hole, board, pot, to_call, can_raise, street,
                                   history=self.history, hero_index=seat)
        if decision == "raise" and can_raise:
            return Poker.BET_RAISE
        if decision == "fold" and Poker.FOLD in env.get_legal_actions():
            return Poker.FOLD
        return Poker.CHECK_CALL

    def apply(self, seat, action):
        action_str = {Poker.FOLD: "fold", Poker.CHECK_CALL: "call", Poker.BET_RAISE: "raise"}[action]
        street_str = Poker.INT2STRING_ROUND[self.env.current_round]
        self.history.append({"actor": seat, "street": street_str, "action": action_str})
        _obs, _rew, done, _info = self.env.step(action)
        self.done = done


G = Game()
app = Flask(__name__)


def state_for(seat: int) -> dict:
    env = G.env
    seats = []
    for p in env.seats:
        reveal = (p.seat_id == seat) or G.done or p.is_allin
        seats.append({
            "seat": p.seat_id,
            "stack": p.stack,
            "current_bet": p.current_bet,
            "folded": p.folded_this_episode,
            "is_allin": p.is_allin,
            "cards": cards_str(p.hand) if reveal else "",
            "n_cards": 4,
            "hand_rank": p.hand_rank if G.done else None,
            "is_bot": (G.mode == "bot" and p.seat_id == BOT_SEAT),
        })
    legal = env.get_legal_actions() if (not G.done and env.current_player.seat_id == seat) else []
    return {
        "mode": G.mode,
        "bot_type": G.bot_type,
        "board": cards_str(env.board),
        "pot": env.main_pot + sum(p.current_bet for p in env.seats),
        "round": Poker.INT2STRING_ROUND.get(env.current_round, "-"),
        "n_raises_this_round": env.n_raises_this_round,
        "small_bet": FixedLimitOmahaHiLo.SMALL_BET,
        "big_bet": FixedLimitOmahaHiLo.BIG_BET,
        "current_player": None if G.done else env.current_player.seat_id,
        "legal_actions": legal,
        "seats": seats,
        "done": G.done,
        "continue_stacks": G.continue_stacks,
        "session_over": G.session_over,
    }


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/api/new_game", methods=["POST"])
def api_new_game():
    body = request.json or {}
    mode = body.get("mode", "bot")
    bot_type = body.get("bot_type", "abc")
    continue_stacks = bool(body.get("continue_stacks", True))
    G.new_game(mode, bot_type, continue_stacks)
    return jsonify({"ok": True})


@app.route("/api/reset_stacks", methods=["POST"])
def api_reset_stacks():
    if G.env is None:
        return jsonify({"error": "No active game"}), 400
    G.new_game(G.mode, G.bot_type or "abc", G.continue_stacks)
    return jsonify({"ok": True})


@app.route("/api/state")
def api_state():
    if G.env is None:
        return jsonify({"error": "No active game"}), 404
    seat = int(request.args.get("seat", 0))
    return jsonify(state_for(seat))


@app.route("/api/action", methods=["POST"])
def api_action():
    if G.env is None or G.done:
        return jsonify({"error": "No hand in progress"}), 400
    body = request.json or {}
    seat = int(body.get("seat", HUMAN_SEAT))
    action = int(body.get("action"))
    if G.env.current_player.seat_id != seat:
        return jsonify({"error": "Not your turn"}), 400
    if action not in G.env.get_legal_actions():
        return jsonify({"error": "Illegal action"}), 400
    G.apply(seat, action)
    return jsonify(state_for(seat))


@app.route("/api/bot_step", methods=["POST"])
def api_bot_step():
    if G.env is None or G.done or G.mode != "bot":
        return jsonify(state_for(HUMAN_SEAT))
    if G.env.current_player.seat_id != BOT_SEAT:
        return jsonify(state_for(HUMAN_SEAT))
    action = G.bot_decision(BOT_SEAT)
    G.apply(BOT_SEAT, action)
    return jsonify(state_for(HUMAN_SEAT))


@app.route("/api/next_hand", methods=["POST"])
def api_next_hand():
    if G.env is None or not G.done:
        return jsonify({"error": "Hand not over"}), 400
    G.next_hand()
    return jsonify({"ok": True})


HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>&#127183;</text></svg>">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no">
<title>Fixed-Limit Omaha Hi/Lo</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%;background:#1a5c2a;color:#fff;font-family:system-ui,-apple-system,sans-serif;-webkit-tap-highlight-color:transparent}
body{display:flex;flex-direction:column;align-items:center;padding:8px;max-width:520px;margin:0 auto;gap:5px}

.hdr{width:100%;display:flex;justify-content:space-between;align-items:center;padding:7px 10px;background:rgba(0,0,0,.38);border-radius:8px}
.hdr h1{font-size:1rem;font-weight:800;letter-spacing:2px;color:#a5d6a7}
.hinfo{font-size:.72rem;opacity:.75;text-align:right;line-height:1.5}

.pbox{width:100%;background:rgba(0,0,0,.22);border-radius:10px;padding:10px 12px;border:2px solid transparent;transition:border-color .2s}
.pbox.actor{border-color:#ffeb3b}
.pbox.human{background:rgba(0,0,0,.32)}
.ph{display:flex;align-items:center;gap:6px;margin-bottom:6px;flex-wrap:wrap}
.pname{font-weight:700;font-size:.92rem;flex:1}
.pstack{font-size:.85rem;color:#ffd54f;font-weight:600}
.plabel{font-size:.67rem;color:#a5d6a7;background:rgba(255,255,255,.12);padding:2px 6px;border-radius:4px;white-space:nowrap}

.cards{display:flex;gap:5px;flex-wrap:wrap;min-height:60px;align-items:center}
.card{width:44px;height:60px;background:#fff;border-radius:6px;display:flex;flex-direction:column;align-items:center;justify-content:center;font-size:1rem;font-weight:800;line-height:1.1;box-shadow:0 2px 6px rgba(0,0,0,.5);color:#111;user-select:none;flex-shrink:0}
.card .s{font-size:1rem;line-height:1}
.card.r{color:#c62828}
.card.hid{background:#1a3a6b;background-image:repeating-linear-gradient(45deg,rgba(255,255,255,.07) 0,rgba(255,255,255,.07) 1px,transparent 1px,transparent 7px)}
.pbet{font-size:.75rem;color:#ff8a65;margin-top:3px}

.board{width:100%;background:rgba(0,0,0,.18);border-radius:10px;padding:10px 14px;text-align:center}
.pot{font-size:1.05rem;font-weight:700;color:#ffd54f;margin-bottom:5px}
.blbl{font-size:.67rem;color:#a5d6a7;text-transform:uppercase;letter-spacing:1px;margin-bottom:5px}
.bcards{display:flex;gap:5px;justify-content:center;flex-wrap:wrap;min-height:60px;align-items:center}
.sbdg{font-size:.66rem;background:rgba(255,255,255,.15);padding:2px 7px;border-radius:10px;margin-left:6px;font-weight:600;vertical-align:middle}

.actarea{width:100%;display:flex;flex-direction:column;gap:7px}
.btnrow{display:flex;gap:7px}
.btn{border:none;border-radius:8px;padding:15px 10px;font-size:.92rem;font-weight:700;cursor:pointer;color:#fff;touch-action:manipulation;flex:1;transition:opacity .1s,transform .08s}
.btn:active{opacity:.72;transform:scale(.96)}
.btn:disabled{opacity:.35;cursor:default}
.fold{background:#b71c1c}.call{background:#1565c0}.rbtn{background:#e65100}
.bnew{background:#2e7d32;width:100%;padding:16px}
.bnxt{background:#4a148c;width:100%;padding:16px}

.res{width:100%;background:rgba(0,0,0,.38);border-radius:10px;padding:14px;text-align:center}
.res h2{font-size:1.05rem;margin-bottom:10px;color:#ffd54f}
.rline{display:flex;justify-content:space-between;padding:5px 0;font-size:.85rem;border-bottom:1px solid rgba(255,255,255,.1)}
.rline.win{color:#86efac}.rline.lose{color:#fca5a5}

.start{width:100%;display:flex;flex-direction:column;gap:14px;padding:16px 0}
.start h2{text-align:center;font-size:1.15rem;color:#a5d6a7}
.fg{display:flex;flex-direction:column;gap:5px}
.fg label{font-size:.82rem;color:#a5d6a7}
.fg select{padding:12px;border-radius:7px;border:none;font-size:1rem;background:rgba(255,255,255,.92);color:#111}
.modebtns{display:flex;gap:8px}
.modebtn{flex:1;padding:14px 8px;border-radius:8px;border:2px solid rgba(255,255,255,.15);background:rgba(0,0,0,.2);color:#fff;font-size:.95rem;font-weight:700;cursor:pointer}
.modebtn.sel{border-color:#ffd54f;background:rgba(255,213,79,.15)}
.friendlink{background:rgba(0,0,0,.3);border-radius:8px;padding:12px;font-size:.85rem;word-break:break-all}

.thinking{text-align:center;padding:14px;opacity:.65;font-size:.88rem}
</style>
</head>
<body>
<div class="hdr">
  <h1>FL OMAHA HI/LO</h1>
  <div class="hinfo" id="hinfo">8-or-Better</div>
</div>
<div id="app" style="width:100%;display:flex;flex-direction:column;gap:5px"></div>

<script>
const params = new URLSearchParams(location.search);
const urlSeat = params.get('seat');
const seat = urlSeat !== null ? parseInt(urlSeat) : 0;
let mode = 'bot', botType = 'abc', continueStacks = true;
let G = null;
let botStepTimer = null;
const $ = id => document.getElementById(id);
const app = $('app');

async function post(url, body) {
  const r = await fetch(url, {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body||{})});
  return r.json();
}

const SUIT = {s:'&#9824;', h:'&#9829;', d:'&#9830;', c:'&#9827;'}, RED = {h:1, d:1};
function cardHtml(rankSuit) {
  const rank = rankSuit.slice(0, -1), suit = rankSuit.slice(-1).toLowerCase();
  return `<div class="card${RED[suit] ? ' r' : ''}">${rank}<span class="s">${SUIT[suit]||suit}</span></div>`;
}
function hiddenCardHtml() { return '<div class="card hid"></div>'; }
function renderCardString(s, hidden, nCards) {
  if (hidden) return Array(nCards).fill(0).map(hiddenCardHtml).join('');
  const cards = [];
  for (let i = 0; i < s.length; i += 2) cards.push(s.slice(i, i+2));
  return cards.map(cardHtml).join('');
}

function playerBox(p, isMe, isActor) {
  const label = isMe ? 'You' : (p.is_bot ? `Bot (${G.bot_type})` : 'Opponent');
  const hidden = p.cards === '';
  let rankStr = '';
  if (p.hand_rank) {
    rankStr = `<div class="pbet">hi=${p.hand_rank[0]}${p.hand_rank[1] !== null ? ', lo=' + p.hand_rank[1] : ' (no low)'}</div>`;
  }
  return `<div class="pbox${isActor ? ' actor' : ''}${isMe ? ' human' : ''}">
    <div class="ph">
      <span class="pname">${label}${p.folded ? ' (folded)' : ''}${p.is_allin ? ' (all-in)' : ''}</span>
      <span class="pstack">$${p.stack}</span>
    </div>
    <div class="cards">${renderCardString(p.cards, hidden, p.n_cards)}</div>
    ${p.current_bet ? `<div class="pbet">Bet: $${p.current_bet}</div>` : ''}
    ${rankStr}
  </div>`;
}

function needsBotStep(s) {
  return s && s.mode === 'bot' && !s.done && s.current_player === 1;
}

async function botStep() {
  const s = await post('/api/bot_step');
  G = s;
  render();
  if (needsBotStep(G)) botStepTimer = setTimeout(botStep, 900);
}

async function newGame() {
  clearTimeout(botStepTimer);
  await post('/api/new_game', {mode, bot_type: botType, continue_stacks: continueStacks});
  await poll();
}

async function nextHand() {
  clearTimeout(botStepTimer);
  await post('/api/next_hand');
  await poll();
}

async function resetStacks() {
  clearTimeout(botStepTimer);
  await post('/api/reset_stacks');
  await poll();
}

async function act(action) {
  clearTimeout(botStepTimer);
  G = await post('/api/action', {seat, action});
  render();
  if (needsBotStep(G)) botStepTimer = setTimeout(botStep, 900);
}

function render() {
  if (!G) { renderStart(); return; }
  const me = G.seats[seat], opp = G.seats[1 - seat];
  $('hinfo').textContent = `FL $${G.small_bet}/$${G.big_bet}  |  ${G.round.toUpperCase()}`;

  let h = '';
  h += playerBox(opp, false, !G.done && G.current_player === opp.seat);
  h += `<div class="board">
    <div class="pot">Pot: $${G.pot}</div>
    <div class="blbl">Board<span class="sbdg">${G.round}${G.n_raises_this_round ? ' &middot; raises: ' + G.n_raises_this_round : ''}</span></div>
    <div class="bcards">${G.board ? renderCardString(G.board, false, 5) : '<span style="opacity:.35">&mdash;</span>'}</div>
  </div>`;
  h += playerBox(me, true, !G.done && G.current_player === me.seat);

  h += '<div class="actarea">';
  if (G.done) {
    h += `<div class="res"><h2>${G.session_over ? 'Session over' : 'Hand over'}</h2>
      <div class="rline"><span>You</span><span>$${me.stack}</span></div>
      <div class="rline"><span>${opp.is_bot ? 'Bot' : 'Opponent'}</span><span>$${opp.stack}</span></div>
    </div>`;
    if (G.session_over) {
      h += `<p style="font-size:.8rem;opacity:.75;text-align:center">Someone's too low on chips to post the next blind.</p>`;
      h += `<button class="btn bnxt" onclick="resetStacks()">Reset Stacks &amp; Continue</button>`;
    } else {
      h += `<button class="btn bnxt" onclick="nextHand()">Next Hand &#9654;</button>`;
      if (G.continue_stacks) {
        h += `<button class="btn" style="background:rgba(255,255,255,.12);font-size:.8rem;padding:10px" onclick="resetStacks()">Reset Stacks</button>`;
      }
    }
  } else if (G.current_player === seat) {
    const btns = [];
    if (G.legal_actions.includes(0)) btns.push(`<button class="btn fold" onclick="act(0)">Fold</button>`);
    if (G.legal_actions.includes(1)) btns.push(`<button class="btn call" onclick="act(1)">${me.current_bet===opp.current_bet && G.round==='preflop' ? 'Check/Call' : 'Call'}</button>`);
    if (G.legal_actions.includes(2)) btns.push(`<button class="btn rbtn" onclick="act(2)">Raise</button>`);
    h += `<div class="btnrow">${btns.join('')}</div>`;
  } else {
    h += `<div class="thinking">${opp.is_bot ? 'Bot is thinking&hellip;' : 'Waiting on opponent&hellip;'}</div>`;
  }
  h += '</div>';
  app.innerHTML = h;
}

function selectMode(m) {
  mode = m;
  renderStart();
}

function renderStart() {
  app.innerHTML = `<div class="start">
    <h2>New Game</h2>
    <div class="modebtns">
      <button class="modebtn${mode==='bot'?' sel':''}" onclick="selectMode('bot')">vs Bot</button>
      <button class="modebtn${mode==='friend'?' sel':''}" onclick="selectMode('friend')">vs Friend</button>
    </div>
    ${mode === 'bot' ? `
    <div class="fg"><label>Opponent</label>
      <select id="bt" onchange="botType=this.value">
        <option value="abc" ${botType==='abc'?'selected':''}>ABCBot &mdash; rule-based hand-strength thresholds</option>
        <option value="bayesian" ${botType==='bayesian'?'selected':''}>BayesianBot &mdash; trained naive-Bayes + opponent modeling</option>
      </select></div>
    ` : `
    <p style="font-size:.85rem;opacity:.8">You'll play seat 0. Share this page's URL with <code>?seat=1</code> appended with your friend.</p>
    <div class="friendlink">${location.origin}${location.pathname}?seat=1</div>
    `}
    <div class="fg" style="flex-direction:row;align-items:center;gap:8px">
      <input type="checkbox" id="cs" ${continueStacks?'checked':''} onchange="continueStacks=this.checked" style="width:18px;height:18px">
      <label for="cs" style="font-size:.85rem">Carry stacks over between hands (uncheck to reset both players to $${STARTING_STACK_JS} every hand)</label>
    </div>
    <button class="btn bnew" onclick="newGame()">Start Game</button>
  </div>`;
}

let noGameYet = false;

async function poll() {
  const r = await fetch(`/api/state?seat=${seat}`);
  if (r.status === 404) {
    G = null;
    noGameYet = true;
    if (seat === 0) { renderStart(); }
    else { app.innerHTML = '<div class="thinking">Waiting for seat 0 to start the game&hellip;</div>'; }
    return;
  }
  noGameYet = false;
  G = await r.json();
  render();
  if (needsBotStep(G) && !botStepTimer) botStepTimer = setTimeout(botStep, 900);
}

poll();
// Once seat 0 has confirmed there's no game yet, stop hammering /api/state --
// nothing changes until they click "Start Game" themselves (which re-polls
// directly). Seat 1 keeps polling while genuinely waiting on the host.
setInterval(() => {
  if (needsBotStep(G)) return;
  if (noGameYet && seat === 0) return;
  poll();
}, 1500);
</script>
</body>
</html>
"""

HTML = HTML.replace("STARTING_STACK_JS", str(STARTING_STACK))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=False)
