"""
pokerlib.py - vectorized (numpy) Short Deck PLO5 hand evaluation and sampling.

Every card gets an additive integer key:
    low 22 bits : 5 ** rank          (sum of 5 cards identifies the rank multiset)
    high bits   : 1 << (3 * suit)    (sum of 5 cards tells whether it is a flush)
So the key of a 5-card hand is just the sum of its cards' keys, and a pair of
hole cards plus a triple of board cards can be combined with one addition.
Scores are identical to evaluator.score_five (validated at import).
"""

from __future__ import annotations

from itertools import combinations, combinations_with_replacement

import numpy as np

from evaluator import NUM_CARDS, score_five

NUM_RANKS = 9
RANK_BASE = 5 ** NUM_RANKS          # 1,953,125: size of the rank-multiset space
SUIT_SHIFT = 22
LOW_MASK = (1 << SUIT_SHIFT) - 1

CARD_KEY = np.array([5 ** (c >> 2) + ((1 << (3 * (c & 3))) << SUIT_SHIFT) for c in range(NUM_CARDS)],
                    dtype=np.int64)

PAIRS5 = np.array(list(combinations(range(5), 2)), dtype=np.int64)    # (10, 2)
TRIPLES5 = np.array(list(combinations(range(5), 3)), dtype=np.int64)  # (10, 3)
TRIPLES4 = np.array(list(combinations(range(4), 3)), dtype=np.int64)  # (4, 3)


def _build_tables() -> tuple[np.ndarray, np.ndarray]:
    score = np.zeros(2 * RANK_BASE, dtype=np.int32)
    for ranks in combinations_with_replacement(range(NUM_RANKS), 5):
        if max(ranks.count(r) for r in ranks) > 4:
            continue
        # Give each repeated rank a different suit, and break flushes for
        # five distinct ranks by moving one card to another suit.
        seen: dict[int, int] = {}
        cards = []
        for r in ranks:
            s = seen.get(r, 0)
            seen[r] = s + 1
            cards.append(r * 4 + s)
        if len({c & 3 for c in cards}) == 1:
            cards[0] = (cards[0] & ~3) | 1
        idx = sum(5 ** r for r in ranks)
        score[idx] = score_five(cards)
        if len(set(ranks)) == 5:
            score[RANK_BASE + idx] = score_five([r * 4 for r in ranks])
    flush_offset = np.zeros(1 << 12, dtype=np.int64)
    for s in range(4):
        flush_offset[5 << (3 * s)] = RANK_BASE
    return score, flush_offset


SCORE, FLUSH_OFFSET = _build_tables()


def lookup(keys: np.ndarray) -> np.ndarray:
    """Scores for an array of 5-card keys."""
    return SCORE[(keys & LOW_MASK) + FLUSH_OFFSET[keys >> SUIT_SHIFT]]


def pair_keys(hole: np.ndarray) -> np.ndarray:
    """hole (..., 5) cards -> (..., 10) keys of every 2-card subset."""
    k = CARD_KEY[hole]
    return k[..., PAIRS5[:, 0]] + k[..., PAIRS5[:, 1]]


def triple_keys(board: np.ndarray) -> np.ndarray:
    """board (..., 5) or (..., 4) or (..., 3) -> keys of every 3-card subset."""
    n = board.shape[-1]
    k = CARD_KEY[board]
    if n == 3:
        return k.sum(axis=-1, keepdims=True)
    t = TRIPLES5 if n == 5 else TRIPLES4
    return k[..., t[:, 0]] + k[..., t[:, 1]] + k[..., t[:, 2]]


def best_scores(pk: np.ndarray, tk: np.ndarray) -> np.ndarray:
    """Best Omaha score per row. pk (N,10) or (10,), tk (N,T) or (T,)."""
    if pk.ndim == 1 and tk.ndim == 1:
        return lookup(pk[:, None] + tk[None, :]).max()
    if pk.ndim == 1:
        pk = pk[None, :]
    if tk.ndim == 1:
        tk = tk[None, :]
    total = pk[:, :, None] + tk[:, None, :]
    return lookup(total).reshape(total.shape[0], -1).max(axis=1)


# ---------------------------------------------------------------------------
# Preflop strength proxy (used to rank opponent hands before the flop).
# A linear model over simple features, fitted offline against Monte Carlo
# equity vs a random hand (see dev/fit_preflop.py).
# ---------------------------------------------------------------------------

PREFLOP_W = np.array([-0.509304, 0.007153, 0.197261, 0.028930, 0.004921, 0.028309, -0.018957,
                      -0.018033, 0.000236, 0.158229, 0.001654, -0.010057, 0.005216, 0.014492])


def preflop_features(hole: np.ndarray) -> np.ndarray:
    """hole (N,5) -> (N,F) feature matrix."""
    ranks = hole >> 2
    suits = hole & 3
    n = hole.shape[0]
    rc = np.zeros((n, NUM_RANKS), dtype=np.int64)
    np.add.at(rc, (np.arange(n)[:, None], ranks), 1)
    sc = np.zeros((n, 4), dtype=np.int64)
    np.add.at(sc, (np.arange(n)[:, None], suits), 1)
    pairs = (rc >= 2).sum(axis=1)
    trips = (rc >= 3).sum(axis=1)
    pair_rank = (np.arange(NUM_RANKS)[None, :] * (rc >= 2)).max(axis=1)
    suited2 = (sc >= 2).sum(axis=1)
    max_suit = sc.max(axis=1)
    # Rank of the best card in each suit that has at least 2 cards.
    suited_hi = np.zeros(n, dtype=np.int64)
    for s in range(4):
        in_s = suits == s
        hi = np.where(in_s, ranks, -1).max(axis=1)
        suited_hi = np.maximum(suited_hi, np.where(sc[:, s] >= 2, hi + 1, 0))
    present = rc > 0
    distinct = present.sum(axis=1)
    # Connectivity: most distinct ranks inside any 5-rank window (with the wheel).
    ext = np.concatenate([present[:, 8:9], present], axis=1).astype(np.int64)  # ace low
    cs = np.cumsum(np.concatenate([np.zeros((n, 1), dtype=np.int64), ext], axis=1), axis=1)
    window = (cs[:, 5:] - cs[:, :-5]).max(axis=1)
    high = ranks.sum(axis=1)
    aces = rc[:, 8]
    return np.stack([
        np.ones(n), high, pairs, trips, pair_rank, suited2, (max_suit >= 3), (max_suit >= 4),
        suited_hi, distinct, window, aces, (window >= 4), (window >= 5),
    ], axis=1).astype(np.float64)


def preflop_strength(hole: np.ndarray) -> np.ndarray:
    return preflop_features(hole) @ PREFLOP_W


# ---------------------------------------------------------------------------
# Monte Carlo sampling
# ---------------------------------------------------------------------------

def sample_unknown(rng: np.random.Generator, dead: list[int], n: int, k: int) -> np.ndarray:
    """n rows of k distinct cards drawn from the deck minus `dead`."""
    mask = np.ones(NUM_CARDS, dtype=bool)
    mask[list(dead)] = False
    deck = np.flatnonzero(mask)
    order = np.argsort(rng.random((n, deck.size)), axis=1)[:, :k]
    return deck[order]


def simulate(rng: np.random.Generator, hole: tuple, board: tuple, n: int) -> dict:
    """Sample n (opponent hand, runout) pairs.

    Returns per-sample arrays:
        result    1 win / 0.5 tie / 0 loss for us at showdown
        opp_str   strength of the opponent's hand on the CURRENT board
                  (made-hand score postflop, preflop model before the flop)
        my_now    our made-hand score on the current board (postflop)
    """
    hole_a = np.array(hole, dtype=np.int64)
    board_a = np.array(board, dtype=np.int64)
    nb = len(board)
    draw = sample_unknown(rng, list(hole) + list(board), n, 5 + (5 - nb))
    opp = draw[:, :5]
    my_pk = pair_keys(hole_a)
    opp_pk = pair_keys(opp)
    if nb == 5:
        tk = triple_keys(board_a)
        mine = best_scores(my_pk, tk)
        theirs = best_scores(opp_pk, tk)
        opp_str = theirs
        my_now = mine
    else:
        full = np.concatenate([np.broadcast_to(board_a, (n, nb)), draw[:, 5:]], axis=1)
        tk = triple_keys(full)
        mine = best_scores(my_pk, tk)
        theirs = best_scores(opp_pk, tk)
        if nb == 0:
            opp_str = preflop_strength(opp)
            my_now = None
        else:
            tk_now = triple_keys(board_a)
            opp_str = best_scores(opp_pk, np.broadcast_to(tk_now, (n, tk_now.size)))
            my_now = best_scores(my_pk, tk_now)
    result = (mine > theirs).astype(np.float64) + 0.5 * (mine == theirs)
    # Tiny jitter so quantile cuts on tied strengths are not all-or-nothing.
    opp_str = opp_str.astype(np.float64) + rng.random(n) * 1e-3
    return {"result": result, "opp_str": opp_str, "my_now": my_now}


def _self_check() -> None:
    from evaluator import omaha_score
    rng = np.random.default_rng(12345)
    for _ in range(300):
        cards = rng.permutation(NUM_CARDS)[:10]
        hole, board = cards[:5], cards[5:]
        fast = int(best_scores(pair_keys(hole), triple_keys(board)))
        if fast != omaha_score(list(hole), list(board)):
            raise AssertionError("pokerlib evaluator disagrees with evaluator.omaha_score")


_self_check()
