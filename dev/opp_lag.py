"""A loose-aggressive equity bot: bets decent hands, bluffs often, calls wide."""

import numpy as np

import pokerlib
from game import Action

BLUFF = 0.35


def new_half(config, memory):
    memory["rng"] = np.random.default_rng(5)


def choose_action(obs, memory):
    rng = memory["rng"]
    eq = float(pokerlib.simulate(rng, obs.hole_cards, obs.board, 300)["result"].mean())
    legal = obs.legal_actions
    can_raise = Action.RAISE_POT in legal
    if can_raise and (eq >= 0.62 or rng.random() < BLUFF):
        return Action.RAISE_POT if rng.random() < 0.5 else Action.RAISE_HALF
    need = obs.to_call / (obs.pot + obs.to_call) if obs.to_call else 0.0
    if eq >= need - 0.05 or obs.to_call == 0:
        return Action.CHECK_CALL
    return Action.FOLD


def on_hand_end(result, memory):
    pass
