"""
agent.py - YOUR poker bot. This one file is a complete web server.

Run it:
    python agent.py                  # listens on port 5008
    PORT=5010 python agent.py        # or pick a port

Play a match against a baseline (three windows):
    python agent.py                                 # your bot on port 5008
    python baseline-bots/baseline_random.py         # opponent on port 5009
    python judge_engine.py                          # judge: both halves (uses those two ports)

Check that the tournament can run it:
    python -m pytest -q              # tests in test_agent.py
    ./tools/check_build.sh           # builds and smoke-tests it like the tournament

TODO:
    1. Assign your agent a unique PARTICIPANT and AGENT_NAME below.
    2. Implement the three functions between "YOUR CODE HERE" and "END OF YOUR CODE".

Everything under "SERVER PLUMBING" talks to the judge. You normally never
need to change it, and if you break it the judge can't talk to your bot.
"""

import os
import traceback

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from evaluator import card_rank, card_suit, cards_to_str, omaha_score, warmup  # handy helpers, see evaluator.py
from game import Action, AgentConfig, HandResult, Observation, Street

# Build omaha_score's lookup table now, while the server starts, instead of
# during your first decision (where it would eat into your time bank).
# Do any other slow setup (loading a model, ...) here too: the server must
# answer within 15 seconds of starting.
warmup()

# rename as your team/bot's name
PARTICIPANT = "TotallyAwesomeDatathonParticipant"   # TODO: your team name
AGENT_NAME = "RangeEV"


# ===== YOUR CODE HERE =====

import time

import numpy as np

import pokerlib

RAISES = (Action.RAISE_HALF, Action.RAISE_POT)

# Priors for the opponent model, as (rate, pseudo-count). Index 0 = preflop, 1 = postflop.
PRIOR_FOLD_TO_RAISE = ((0.30, 8.0), (0.40, 8.0))
PRIOR_RAISE_RATE = ((0.30, 8.0), (0.30, 8.0))
# Share of the opponent's shown-down raises made with a below-median hand.
PRIOR_WEAK_RAISE = ((0.10, 6.0), (0.10, 6.0))
SHOWDOWN_SAMPLES = 150
RANGE_DECAY = 0.6   # how much earlier streets' actions still narrow the range

# Monte Carlo sample counts. Big pots get more samples; everything is cut back
# as the (estimated) time bank charge grows.
SAMPLES_FULL = 400
SAMPLES_BIG_POT = 900
BIG_POT = 16
# The judge charges each request's time beyond 1 ms. We only see our own
# compute time, so assume the rest of the request costs this much more
# (a slow tournament machine) when estimating the charge.
OVERHEAD_FACTOR = 1.8
GRACE = 0.001
TIME_SOFT = 3.0      # estimated seconds charged before sampling less
TIME_HARD = 5.0      # ... before sampling much less


def _warm_numpy() -> None:
    rng = np.random.default_rng(0)
    for board in ((), (0, 5, 10), (0, 5, 10, 15), (0, 5, 10, 15, 20)):
        pokerlib.simulate(rng, (30, 31, 25, 22, 17), board, SAMPLES_FULL)


_warm_numpy()


def new_half(config: AgentConfig, memory: dict) -> None:
    """Called at the start of each half-match (1000 hands).

    `memory` is a fresh, empty dict every half. Store anything you want to
    remember between hands in it, e.g. memory["opp_raises"] = 0.
    `config` holds the match settings (starting_stack, big_blind, ...).

    In the tournament, each half runs in a brand-new copy of your server,
    so nothing you learn in one half carries over to the other half.
    """
    memory["hands_played"] = 0
    memory["rng"] = np.random.default_rng(int(time.time_ns() % (2 ** 32)))
    memory["time_used"] = 0.0
    # [phase] -> [faced our raise, folded to it]
    memory["fold_stats"] = [[0, 0], [0, 0]]
    # [phase] -> [opponent actions, of which raises]
    memory["raise_stats"] = [[0, 0], [0, 0]]
    # [phase] -> [opponent raises seen at showdown, of which made with a below-median hand]
    memory["bluff_stats"] = [[0, 0], [0, 0]]
    memory["opp_raise_streets"] = []  # streets of the opponent's raises this hand
    memory["cur_hand"] = -1       # hand index of the last observation
    memory["hist_seen"] = 0       # action_history entries already counted
    memory["prev_entry"] = None   # (street, is_me, action) of the last counted entry
    memory["my_last"] = None      # (street, action) we returned last
    memory["sim_key"] = None
    memory["sim"] = None


def _phase(street) -> int:
    return 0 if street == Street.PREFLOP else 1


def _rate(stats: list, prior: tuple) -> float:
    base, weight = prior
    return (stats[1] + base * weight) / (stats[0] + weight)


def _count_opp_action(memory: dict, street, action, prev) -> None:
    ph = _phase(street)
    rs = memory["raise_stats"][ph]
    rs[0] += 1
    rs[1] += action in RAISES
    if prev is not None and prev[1] and prev[2] in RAISES and prev[0] == street:
        fs = memory["fold_stats"][ph]
        fs[0] += 1
        fs[1] += action == Action.FOLD


def _update_from_history(obs: Observation, memory: dict) -> None:
    if obs.hand_index != memory["cur_hand"]:
        memory["cur_hand"] = obs.hand_index
        memory["hist_seen"] = 0
        memory["prev_entry"] = None
        memory["opp_raise_streets"] = []
    hist = obs.action_history
    for entry in hist[memory["hist_seen"]:]:
        if not entry.is_me:
            _count_opp_action(memory, entry.street, entry.action, memory["prev_entry"])
            if entry.action in RAISES:
                memory["opp_raise_streets"].append(entry.street)
        memory["prev_entry"] = (entry.street, entry.is_me, entry.action)
    memory["hist_seen"] = len(hist)


def _bluff_share(memory: dict, ph: int) -> float:
    """Estimated share of the opponent's raises made with below-median hands
    beyond what honest raising would show (random hands are below median half the time)."""
    return min(0.9, 2.0 * _rate(memory["bluff_stats"][ph], PRIOR_WEAK_RAISE[ph]))


def _range_weights(obs: Observation, memory: dict, pct: np.ndarray) -> np.ndarray:
    """Relative likelihood of each sampled opponent hand given their actions this hand.

    pct is each sample's strength percentile among random hands on the current
    street. A raise is modelled as: always with the top t of hands, and as a
    bluff with a fraction q of the rest, where t and q reproduce the observed
    raise rate and bluff share. Calling our raise keeps the top (1 - fold rate).
    """
    per_street: dict = {}
    prev = None
    for entry in obs.action_history:
        if not entry.is_me:
            s = per_street.setdefault(entry.street, [1.0, 1.0, False])
            ph = _phase(entry.street)
            if entry.action in RAISES:
                r = min(0.95, max(0.08, _rate(memory["raise_stats"][ph], PRIOR_RAISE_RATE[ph])))
                b = _bluff_share(memory, ph)
                t = r * (1.0 - b)
                q = r * b / (1.0 - t)
                s[0] *= t
                s[1] *= q
            elif prev is not None and prev.is_me and prev.action in RAISES and prev.street == entry.street:
                f = min(0.75, max(0.0, _rate(memory["fold_stats"][ph], PRIOR_FOLD_TO_RAISE[ph])))
                s[0] *= 1.0 - f
                s[2] = True
        prev = entry
    w = np.ones(pct.size)
    for street, (top, q, called) in per_street.items():
        if top >= 0.999:
            continue
        floor = max(q, 0.12 if called else 0.0, 0.01)
        decay = 1.0 if street == obs.street else RANGE_DECAY
        w *= np.where(pct >= 1.0 - top, 1.0, floor) ** decay
    return w


def _calling_part(strength: np.ndarray, w: np.ndarray, keep: float) -> np.ndarray:
    """Weights restricted to the strongest `keep` share of the weighted range."""
    if keep >= 0.999:
        return w
    order = np.argsort(-strength)
    cum = np.cumsum(w[order])
    inside = np.zeros(w.size, dtype=bool)
    inside[order] = (cum - w[order]) < keep * cum[-1]
    return np.where(inside, w, 0.0)


def _get_sim(obs: Observation, memory: dict) -> dict:
    key = (obs.hand_index, int(obs.street))
    used = memory["time_used"]
    if used < TIME_SOFT:
        n = SAMPLES_BIG_POT if obs.pot + obs.to_call >= BIG_POT else SAMPLES_FULL
    else:
        n = 200 if used < TIME_HARD else 80
    cached = memory["sim"]
    if memory["sim_key"] == key and cached is not None and cached["result"].size >= n:
        return cached
    sim = pokerlib.simulate(memory["rng"], obs.hole_cards, obs.board, n)
    sim["pct"] = np.argsort(np.argsort(sim["opp_str"])) / float(n)
    memory["sim_key"] = key
    memory["sim"] = sim
    return sim


def _decide(obs: Observation, memory: dict) -> Action:
    legal = obs.legal_actions
    sim = _get_sim(obs, memory)
    result, strength, pct = sim["result"], sim["opp_str"], sim["pct"]
    w = _range_weights(obs, memory, pct)
    eq = float((result * w).sum() / w.sum())

    pot, to_call = obs.pot, obs.to_call
    evs = {Action.CHECK_CALL: eq * (pot + to_call) - to_call}
    if Action.FOLD in legal:
        evs[Action.FOLD] = 0.0

    ph = _phase(obs.street)
    f = _rate(memory["fold_stats"][ph], PRIOR_FOLD_TO_RAISE[ph])
    wc = _calling_part(strength, w, 1.0 - f)
    eq_c = float((result * wc).sum() / wc.sum())
    for a in RAISES:
        if a not in legal:
            continue
        rt = obs.raise_to[a]
        mine = rt - obs.my_bet
        theirs = rt - obs.opp_bet
        evs[a] = f * pot + (1.0 - f) * (eq_c * (pot + mine + theirs) - mine)

    return max(evs, key=evs.get)


def choose_action(obs: Observation, memory: dict) -> Action:
    """Called every time it is your turn. Return one of obs.legal_actions.

    What you can see in obs:
        hand_index          which hand of the half this is (0, 1, 2, ...)
        street              Street.PREFLOP, Street.FLOP, Street.TURN or Street.RIVER
        hole_cards          your 5 cards as ints, sorted ascending (0-35)
        board               board (community) cards seen so far (0, 3, 4 or 5 ints)
        pot                 total chips both players put in this hand
        to_call             chips you must add to match the opponent
        my_bet, opp_bet     chips you,opponent put in on THIS street
        my_stack, opp_stack chips each player still has behind
        is_button           True if you are the button (small blind)
        legal_actions       the Actions you may return right now
        raise_to            {RAISE_HALF: total, RAISE_POT: total} for legal raises
        min_raise_to        smallest normal raise total (all-ins may be lower)
        raises_this_street  raises so far this street (max 4)
        action_history      this hand's actions so far; each has
                            street, is_me, action, amount, street_total
        big_blind           size of the big blind in chips
    """
    start = time.perf_counter()
    action = Action.CHECK_CALL
    try:
        _update_from_history(obs, memory)
        action = _decide(obs, memory)
    except Exception:
        traceback.print_exc()
    if action not in obs.legal_actions:
        action = Action.CHECK_CALL
    memory["my_last"] = (obs.street, action)
    spent = (time.perf_counter() - start) * OVERHEAD_FACTOR
    memory["time_used"] = memory.get("time_used", 0.0) + max(0.0, spent - GRACE)
    return action


BOARD_VISIBLE = {Street.PREFLOP: 0, Street.FLOP: 3, Street.TURN: 4, Street.RIVER: 5}


def _learn_from_showdown(result: HandResult, memory: dict) -> None:
    """For each raise the opponent made this hand, check whether the hand
    they showed was below median (equity < 50% vs a random hand) at the time."""
    if memory["time_used"] > TIME_SOFT:
        return
    for street in set(memory["opp_raise_streets"]):
        board = tuple(result.board[:BOARD_VISIBLE[street]])
        dead_hole = tuple(result.opp_hole)
        eq = float(pokerlib.simulate(memory["rng"], dead_hole, board, SHOWDOWN_SAMPLES)["result"].mean())
        bs = memory["bluff_stats"][_phase(street)]
        bs[0] += 1
        bs[1] += eq < 0.5


def on_hand_end(result: HandResult, memory: dict) -> None:
    """Called once for every finished hand. Optional: learn from what happened.

    To save time, the judge sends finished hands along with your next
    decision request, so this always runs BEFORE choose_action() sees the
    next hand. (The last hands of a half arrive at the end of the half.)

    result has: hand_index, my_net (chips won or lost), pot,
    went_to_showdown, board (cards revealed), opp_hole (opponent's cards,
    only after a showdown, else None), winner ('me', 'opp' or 'split').
    """
    memory["hands_played"] += 1
    start = time.perf_counter()
    try:
        _hand_end(result, memory)
    except Exception:
        traceback.print_exc()
    memory["time_used"] += (time.perf_counter() - start) * OVERHEAD_FACTOR


def _hand_end(result: HandResult, memory: dict) -> None:
    if result.hand_index != memory["cur_hand"]:
        # We never got to act, so the opponent folded on the button.
        if not result.went_to_showdown:
            memory["raise_stats"][0][0] += 1
        return
    if result.opp_hole is not None and len(result.board) == 5:
        _learn_from_showdown(result, memory)
    last = memory["my_last"]
    if last is None or last[1] == Action.FOLD:
        return
    street, action = last
    # The only action after our last decision is the opponent's closing one.
    if action in RAISES:
        folded = not result.went_to_showdown and result.winner == "me"
        _count_opp_action(memory, street, Action.FOLD if folded else Action.CHECK_CALL,
                          (street, True, action))

# ===== END OF YOUR CODE =====


# ===== SERVER PLUMBING (you normally don't need to edit below) =====
# The judge calls these routes:
#   GET  /           -> who you are (also used as the health check)
#   POST /new-half   -> start a fresh half: memory is wiped
#   POST /act        -> the judge sends any finished hands plus an Observation;
#                       on_hand_end() runs for each hand, then you answer an action
#   POST /hand-end   -> the judge sends the last finished hands of a half
#
# Why "async def"? The routes run on one thread and call your functions
# directly, so your functions never run twice at once. Your own functions
# stay normal "def" (don't make them async).
#
# Keep your per-half state in `memory`. In the tournament every half runs in
# a fresh server process, so everything starts over. When you test locally,
# though, one process can play both halves and only `memory` is wiped, NOT
# other global variables. Using `memory` makes your bot behave the same way
# in both places.

app = FastAPI()
MEMORY: dict = {}                                   # your memory for this half
STATUS = {"started": False}                         # has /new-half been called?
TOKEN = os.environ.get("AGENT_TOKEN") or None       # set by the tournament


async def _read_request(request: Request, path: str) -> tuple[dict, JSONResponse | None]:
    """Check the token (and that the half has started) BEFORE reading the body.

    Checking first means a request without the right token costs your bot
    almost nothing, so nobody else can slow it down by sending junk.
    Returns (body, None) when the request is fine, or ({}, error response).
    """
    if TOKEN and request.headers.get("x-agent-token") != TOKEN:
        return {}, JSONResponse({"error": "bad or missing X-Agent-Token"}, status_code=401)
    if not STATUS["started"] and path != "/new-half":
        return {}, JSONResponse({"error": "call /new-half first"}, status_code=409)
    try:
        body = await request.json()
    except ValueError:
        body = None
    if not isinstance(body, dict):
        return {}, JSONResponse({"error": "the request body must be a JSON object"}, status_code=400)
    return body, None


def _error_info(exc: Exception) -> dict:
    """Describe an exception from your code (call this inside `except`)."""
    return {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}


def _crash(exc: Exception, hand_end_errors: list | None = None) -> JSONResponse:
    """Report an exception from your code to the judge (with traceback)."""
    return JSONResponse({**_error_info(exc), "hand_end_errors": hand_end_errors or []},
                        status_code=500)


def _deliver_results(raw_results) -> list:
    """Call on_hand_end() for each finished hand the judge sent.
    Returns a list of error reports (one per hand that crashed)."""
    errors = []
    for raw in raw_results or []:
        try:
            on_hand_end(HandResult.from_dict(raw), MEMORY)
        except Exception as exc:
            errors.append(_error_info(exc))
    return errors


@app.get("/")
async def info():
    """Health check: tells the judge who this bot is."""
    return {"participant": PARTICIPANT, "agent_name": AGENT_NAME}


@app.post("/new-half")
async def route_new_half(request: Request):
    """Wipe memory and call your new_half(). The body is {"config": {...}}."""
    body, refused = await _read_request(request, "/new-half")
    if refused:
        return refused
    MEMORY.clear()
    STATUS["started"] = True
    try:
        new_half(AgentConfig.from_dict(body.get("config", {})), MEMORY)
    except Exception as exc:
        return _crash(exc)
    return {"ok": True}


@app.post("/act")
async def route_act(request: Request):
    """Deliver finished hands, then turn the JSON into an Observation, call
    choose_action(), and send the answer back."""
    body, refused = await _read_request(request, "/act")
    if refused:
        return refused
    hand_end_errors = _deliver_results(body.get("hand_results"))
    try:
        obs = Observation.from_dict(body["observation"])
        action = choose_action(obs, MEMORY)
    except Exception as exc:
        return _crash(exc, hand_end_errors)
    # Send the action's name; anything strange is passed on for the judge to reject.
    if isinstance(action, Action):
        answer = action.name
    elif action is None or isinstance(action, (str, int, float, bool)):
        answer = action
    else:
        answer = repr(action)
    return {"action": answer, "hand_end_errors": hand_end_errors}


@app.post("/hand-end")
async def route_hand_end(request: Request):
    """Call on_hand_end() for the last finished hands of a half."""
    body, refused = await _read_request(request, "/hand-end")
    if refused:
        return refused
    return {"ok": True, "hand_end_errors": _deliver_results(body.get("results"))}


if __name__ == "__main__":
    # The tournament sets PORT as an environment variable, so each bot is compliant to our evaluation platform
    # Locally it defaults to 5008 for your own testing convenience
    port = int(os.environ.get("PORT", "5008"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info", access_log=False)
