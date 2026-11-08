"""A straightforward equity bot: Monte Carlo equity vs a random hand, raise
strong hands, call when the price is right, never bluff, no opponent model."""

import numpy as np

import pokerlib
from game import Action

RAISE_EQ = 0.68
POT_EQ = 0.80


def new_half(config, memory):
    memory["rng"] = np.random.default_rng(99)


def choose_action(obs, memory):
    eq = float(pokerlib.simulate(memory["rng"], obs.hole_cards, obs.board, 300)["result"].mean())
    legal = obs.legal_actions
    if eq >= POT_EQ and Action.RAISE_POT in legal:
        return Action.RAISE_POT
    if eq >= RAISE_EQ and Action.RAISE_HALF in legal:
        return Action.RAISE_HALF
    need = obs.to_call / (obs.pot + obs.to_call) if obs.to_call else 0.0
    if eq >= need + 0.05 or obs.to_call == 0:
        return Action.CHECK_CALL
    return Action.FOLD


def on_hand_end(result, memory):
    pass
