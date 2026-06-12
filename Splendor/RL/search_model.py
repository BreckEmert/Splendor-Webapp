# Splendor/RL/search_model.py
"""
PUCT search (MCTS) at decision time for the webapp - pure numpy, no TF.

SearchModel is a drop-in replacement for InferenceModel: it exposes
get_predictions(state, legal_mask), and player.choose_move's argmax over the
returned vector selects the move. With sims=0 it returns the policy net's
masked logits (greedy play, identical to InferenceModel). With sims>0 it runs
a PUCT tree search over the real game engine and returns the root visit
counts instead - so argmax picks the most-visited (searched) move.

How a move is chosen with sims>0: the live game is snapshotted; each
simulation restores a scratch GUIGame from that snapshot, RESHUFFLES the
scratch decks (deck order is hidden from both players, so search never peeks
at the true draws), then walks the move tree balancing how good each move has
looked (backed-up value), how much the policy net likes it (prior), and how
unexplored it is. New positions are evaluated by the value net; finished games
use the true result. Values flip sign each ply (your win is my loss).

Nets: policy = the shipped inference_model.npz; value = inference_critic.npz
(a Q-vector head; leaf value = tanh(sum(pi * Q) / value_scale)).

The live game object must be attached once after construction:
    model.bind(game)
choose_move only passes (state, mask), and search needs the actual game.
"""

import random

import numpy as np

from Splendor.RL.inference_model import InferenceModel


# --------------------------------------------------------------------------- #
# Snapshot / restore of GUIGame (Card and Noble objects are immutable data:
# containers are copied, card references shared)
# --------------------------------------------------------------------------- #
def snapshot(game):
    b = game.board
    return {
        'gems': b.gems.copy(),
        'cards': [list(tier) for tier in b.cards],
        'decks': [list(d.cards) for d in b.decks],
        'nobles': list(b.nobles),
        'players': [{
            'gems': p.gems.copy(),
            'cards': p.cards.copy(),
            'reserved': list(p.reserved_cards),
            'card_ids': [list(x) for x in p.card_ids],
            'noble_ids': list(p.noble_ids),
            'points': p.points,
            'victor': p.victor,
        } for p in game.players],
        'start_idx': game.start_idx,
        'half_turns': game.half_turns,
        'victor': game.victor,
    }


def restore(game, snap):
    b = game.board
    b.gems = snap['gems'].copy()
    b.cards = [list(tier) for tier in snap['cards']]
    for d, saved in zip(b.decks, snap['decks']):
        d.cards = list(saved)
    b.nobles = list(snap['nobles'])
    for p, ps in zip(game.players, snap['players']):
        p.gems = ps['gems'].copy()
        p.cards = ps['cards'].copy()
        p.reserved_cards = list(ps['reserved'])
        p.card_ids = [list(x) for x in ps['card_ids']]
        p.noble_ids = list(ps['noble_ids'])
        p.points = ps['points']
        p.victor = ps['victor']
    game.start_idx = snap['start_idx']
    game.half_turns = snap['half_turns']
    game.victor = snap['victor']


class _Node:
    __slots__ = ('P', 'N', 'W', 'legal')

    def __init__(self, priors, legal):
        self.P = priors
        self.N = np.zeros_like(priors)
        self.W = np.zeros_like(priors)
        self.legal = legal


class SearchModel:
    """InferenceModel-compatible agent that searches before moving."""

    def __init__(self, policy_path, value_path, sims=40, c_puct=2.0,
                 value_scale=5.0, max_half_turns=300, seed=None):
        self.policy = InferenceModel(policy_path)
        self.value = InferenceModel(value_path)
        self.sims = sims
        self.c_puct = c_puct
        self.value_scale = value_scale
        self.max_half_turns = max_half_turns
        self._rng = random.Random(seed)
        self.game = None          # bound via bind() after game construction
        self._sim = None          # scratch game, built lazily from the class

    def bind(self, game):
        self.game = game
        self._sim = type(game)([('S0', None, 0), ('S1', None, 1)], None)

    # ------------------------------------------------------------------ #
    def get_predictions(self, state, legal_mask):
        """sims=0: masked policy logits (greedy, same as InferenceModel).
        sims>0: root visit counts (argmax => most-visited searched move)."""
        if self.sims <= 0 or self.game is None:
            return self.policy.get_predictions(state, legal_mask)

        visits = self._search(self.game)
        out = visits.astype(np.float32)
        out[~legal_mask] = -np.inf
        return out

    # ------------------------------------------------------------------ #
    def _priors(self, logits, legal):
        z = np.where(legal, logits, -np.inf)
        z = z - z.max()
        e = np.exp(z, where=np.isfinite(z), out=np.zeros_like(z))
        return e / e.sum()

    def _evaluate(self, game):
        """(priors, legal, value in [-1,1]) for the player to move."""
        state = game.to_state()
        legal = game.active_player.get_legal_moves(game.board)
        logits = self.policy._forward(state)
        q = self.value._forward(state)
        priors = self._priors(logits, legal)
        v = float(np.tanh((priors * q).sum() / self.value_scale))
        return priors, legal, v

    def _search(self, game):
        root_snap = snapshot(game)
        tree = {}

        restore(self._sim, root_snap)
        priors, legal, _ = self._evaluate(self._sim)
        tree[()] = _Node(priors, legal)

        for _ in range(self.sims):
            restore(self._sim, root_snap)
            for d in self._sim.board.decks:        # determinize hidden order
                self._rng.shuffle(d.cards)

            path = ()
            edges = []
            v = None
            while True:
                node = tree[path]
                cur_legal = self._sim.active_player.get_legal_moves(
                    self._sim.board)
                avail = node.legal & cur_legal
                if not avail.any():
                    avail = cur_legal
                a = self._select(node, avail)
                edges.append((node, a))

                self._sim.apply_ai_move(a)
                self._sim.half_turns += 1

                if self._sim.victor:
                    v = -1.0                        # mover won; to-move lost
                    break
                if self._sim.half_turns >= self.max_half_turns:
                    *_, v = self._evaluate(self._sim)
                    break

                path = path + (a,)
                if path not in tree:
                    priors, legal, v = self._evaluate(self._sim)
                    tree[path] = _Node(priors, legal)
                    break

            w = v                                   # negamax backup
            for node, a in reversed(edges):
                w = -w
                node.W[a] += w
                node.N[a] += 1

        return tree[()].N

    def _select(self, node, avail):
        idx = np.flatnonzero(avail)
        n, w, p = node.N[idx], node.W[idx], node.P[idx]
        q = np.where(n > 0, w / np.maximum(n, 1), 0.0)
        u = self.c_puct * p * np.sqrt(node.N.sum() + 1.0) / (1.0 + n)
        return int(idx[np.argmax(q + u)])
