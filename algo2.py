from __future__ import annotations
import numpy as np

DISPLAY_NAME = "UCB1 (c=2.0)"
DESCRIPTION  = "Optimistic bonus for less tried arms, balances explore/exploit."

C = 2.0
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
    # Try any unpulled arm first
    zero_idxs = np.where(action_counts == 0)[0]
    if len(zero_idxs) > 0:
        return int(rng.choice(zero_idxs))
    # UCB1 score
    # t can be 0 at the very start; but zero_idxs would have caught it.
    t_eff = max(1, t)
    bonus = C * np.sqrt(np.log(t_eff) / action_counts)
    ucb = q_estimates + bonus
    max_v = np.max(ucb)
    ties = np.where(ucb == max_v)[0]
    return int(rng.choice(ties))
