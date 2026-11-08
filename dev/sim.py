"""In-process duplicate match between two bot modules (no HTTP), for fast testing.

    python dev/sim.py agent.py baseline-bots/baseline_random.py --deals 1000 --seed 1

A bot spec is a path to a .py file with new_half / choose_action / on_hand_end,
optionally followed by ":attr=value;..." to override module globals, e.g.
    "agent.py:SAMPLES_FULL=200;BIG_POT=20"
"""

import argparse
import importlib.util
import math
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from game import Action, AgentConfig, Game, button_for_deal, make_deal  # noqa: E402

_COUNTER = [0]


def load_bot(spec: str):
    path, _, overrides = spec.partition(":")
    if not os.path.isabs(path):
        path = os.path.join(ROOT, path)
    _COUNTER[0] += 1
    name = f"bot_{_COUNTER[0]}"
    s = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(s)
    s.loader.exec_module(mod)
    if overrides:
        for kv in overrides.split(";"):
            k, v = kv.split("=", 1)
            setattr(mod, k, eval(v, vars(mod)))
    mod._label = spec
    return mod


class Seat:
    def __init__(self, mod):
        self.mod = mod
        self.memory = {}
        self.time = 0.0
        self.errors = 0
        self.calls = 0
        self.max_call = 0.0

    def start(self):
        self.memory = {}
        self.mod.new_half(AgentConfig(), self.memory)

    def act(self, obs):
        t = time.perf_counter()
        try:
            a = self.mod.choose_action(obs, self.memory)
        except Exception as exc:  # noqa: BLE001
            self.errors += 1
            print("ERROR", self.mod._label, repr(exc))
            a = None
        dt = time.perf_counter() - t
        self.time += dt
        self.max_call = max(self.max_call, dt)
        self.calls += 1
        if a not in obs.legal_actions:
            if a is not None:
                self.errors += 1
                print("ILLEGAL", self.mod._label, a, obs.legal_actions)
            a = Action.CHECK_CALL if obs.to_call == 0 else Action.FOLD
        return a

    def end(self, res):
        t = time.perf_counter()
        try:
            self.mod.on_hand_end(res, self.memory)
        except Exception as exc:  # noqa: BLE001
            self.errors += 1
            print("ERROR on_hand_end", self.mod._label, repr(exc))
        self.time += time.perf_counter() - t


def play_half(a, b, seed, half, deals):
    """Returns list of a's net chips per deal. In half 1 bot a sits in seat 0."""
    seats = [Seat(a), Seat(b)] if half == 1 else [Seat(b), Seat(a)]
    a_seat = 0 if half == 1 else 1
    for s in seats:
        s.start()
    nets = []
    for d in range(deals):
        g = Game(make_deal(seed, d), button_seat=button_for_deal(d), hand_index=d)
        while not g.is_terminal:
            seat = g.to_act
            g.apply(seats[seat].act(g.observation(seat)))
        for s in (0, 1):
            seats[s].end(g.hand_result(s))
        nets.append(g.result().nets[a_seat])
    return nets, seats[a_seat], seats[1 - a_seat]


def match(a, b, seed, deals, verbose=True):
    n1, sa1, sb1 = play_half(a, b, seed, 1, deals)
    n2, sa2, sb2 = play_half(a, b, seed, 2, deals)
    per = [(x + y) / 2.0 for x, y in zip(n1, n2)]  # bb per deal (chips/2 summed over two hands -> /2)
    mean = sum(per) / len(per)
    var = sum((p - mean) ** 2 for p in per) / (len(per) - 1)
    se = math.sqrt(var / len(per))
    lo, hi = mean - 1.96 * se, mean + 1.96 * se
    verdict = "WIN" if lo > 0 else ("LOSS" if hi < 0 else "CHOP")
    if verbose:
        print(f"{a._label} vs {b._label}: {mean:+.3f} bb/deal  CI [{lo:+.3f}, {hi:+.3f}]  {verdict}"
              f"  | time A {sa1.time:.2f}s/{sa2.time:.2f}s max {max(sa1.max_call, sa2.max_call)*1000:.1f}ms"
              f" calls {sa1.calls}/{sa2.calls} errors {sa1.errors + sa2.errors}"
              f" | B errors {sb1.errors + sb2.errors}")
    return mean, lo, hi


def main():
    p = argparse.ArgumentParser()
    p.add_argument("a")
    p.add_argument("b", nargs="+")
    p.add_argument("--deals", type=int, default=1000)
    p.add_argument("--seed", default="1")
    p.add_argument("--seeds", type=int, default=1)
    args = p.parse_args()
    a = load_bot(args.a)
    for spec in args.b:
        b = load_bot(spec)
        for k in range(args.seeds):
            match(a, b, f"{args.seed}-{k}", args.deals)


if __name__ == "__main__":
    main()
