from __future__ import annotations
import numpy as np

DISPLAY_NAME = "ε-greedy (0.10)"
DESCRIPTION  = "Explore 10% at random otherwise pick highest sample mean."

EPSILON = 0.10
_internal = {"init": False, "num_actions": 0}

def initialize(num_actions: int, rng: np.random.Generator) -> None:
    _internal["num_actions"] = int(num_actions)
    _internal["init"] = True

def choose_action(
    t: int,
    num_actions: int,
    q_estimates: np.ndarray,
    action_counts: np.ndarray,
    history: list[tuple[int, int, float]],
    rng: np.random.Generator,
) -> int:
    if not _internal.get("init", False):
        initialize(num_actions, rng)
    if rng.random() < EPSILON:
        return int(rng.integers(0, num_actions))
    max_q = np.max(q_estimates)
    ties = np.where(q_estimates == max_q)[0]
    return int(rng.choice(ties))
