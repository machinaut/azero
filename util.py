#!/usr/bin/env python

from random import choice
from itertools import compress


def select(valid):
    ''' Select a random valid action '''
    return choice(list(compress(range(len(valid)), valid)))


def argmax(probs, valid):
    ''' Sample a weighted valid action '''
    return max(range(len(probs)), key=lambda x: probs[x] * valid[x])  # noqa


def state2obs(state: State, player: int) -> np.ndarray:
    full_state = state + (player,)
    obs = np.array(full_state, dtype=float)
    return obs


def masked_softmax(x: np.ndarray, mask: Valid) -> np.ndarray:
    e = cast(np.ndarray, np.exp(x - x.max()) * mask)
    s = cast(float, e.sum())
    p = cast(np.ndarray, e / s)
    return p


def sample(logits: np.ndarray, valid: Valid) -> int:
    probs = masked_softmax(logits, valid).tolist()
    return random.choices(range(len(probs)), weights=probs)[0]
