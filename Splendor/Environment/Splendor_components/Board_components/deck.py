# Splendor/Environment/Splendor_components/Board_components/deck.py

import numpy as np
from random import SystemRandom

from .splendor_cards_data import PRELOADED_CARD_DATA


class Card:
    def __init__(self, id, tier, gem, points, cost):
        self.id: int = id
        self.tier: int = tier
        self.gem: int = gem
        self.gem_one_hot: np.ndarray = self.gem_to_one_hot(gem)
        self.points: float = points
        self.cost: np.ndarray = np.concatenate((cost, [0]))  # gem costs

    def gem_to_one_hot(self, index):
        one_hot = np.zeros(5, dtype=int)
        one_hot[index] = 1
        return one_hot

    def to_vector(self, effective_gems):
        # Subtracting helps the model learn how far it is from buying
        clipped_cost = np.maximum(self.cost - effective_gems, 0)[:5]
        return np.concatenate((self.gem_one_hot, [self.points/15], clipped_cost/4))
    

class Noble:
    def __init__(self, id, tier, gem, points, cost):
        self.id: int = id
        self.points: int = points  # always 3
        self.cost: np.ndarray = np.concatenate((cost, [0]))  # visit gem requirement
    
    def to_vector(self, effective_gems) -> np.ndarray:
        # Subtracting helps the model learn how far it is from the Noble
        relative_cost = np.maximum(self.cost - effective_gems, 0)[:5]
        return np.concatenate(([self.points], relative_cost)) / 4.0


_PRELOADED_DECKS = None
def _preload_decks():
    # Read each tier's sheet into memory
    preloaded_decks = {}
    for tier in [0, 1, 2]:
        cards = [
            Card(id=row["id"], tier=tier, gem=row["gem"],
                points=row["points"], cost=row["cost"])
            for row in PRELOADED_CARD_DATA[tier]
        ]
        preloaded_decks[tier] = cards
    
    preloaded_decks['Noble'] = [
        Noble(id=row["id"], tier=tier, gem=row["gem"],
              points=row["points"], cost=row["cost"])
        for row in PRELOADED_CARD_DATA['Noble']
    ]

    return preloaded_decks


class Deck:
    def __init__(self, tier):
        global _PRELOADED_DECKS
        if _PRELOADED_DECKS is None:
            _PRELOADED_DECKS = _preload_decks()

        self.tier = tier

        self.cards = list(_PRELOADED_DECKS[self.tier])
        SystemRandom().shuffle(self.cards)

    def draw(self):
        return self.cards.pop() if self.cards else None
