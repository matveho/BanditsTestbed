"""
Implement your own bandits algorithm here.

Two functions you may edit:

1) initialize(num_actions, rng)
   - Called whenever the app starts Algorithm mode or resets.
   - Use it to reset any internal algorithm state.
   - rng: NumPy generator for randomness.

2) choose_action(t, num_actions, q_estimates, action_counts, history, rng) -> int
   - Return the index of the arm you want to pull at time t (0-based).
   - q_estimates: current sample means per arm (kept by the app)
   - action_counts: how many times each arm was pulled (kept by the app).
   - history: a list of tuples (t, action, reward) for all previous steps.
   - rng: NumPy generator for randomness.

Example policy below is epsilon-greedy with a small epsilon.
"""

import numpy as np

EPSILON = 0.10  # explore 10% of the time

# Internal state example:
_internal = {
    "num_actions": 0,
    "initialized": False,
}

def initialize(num_actions: int, rng: np.random.Generator) -> None:
    _internal["num_actions"] = num_actions
    _internal["initialized"] = True
    # You can initialize your own internal estimates here if you want.
    # Example (commented): _internal["my_estimates"] = np.zeros(num_actions, dtype=float)

def choose_action(
    t: int,
    num_actions: int,
    q_estimates: np.ndarray,
    action_counts: np.ndarray,
    history: list[tuple[int, int, float]],
    rng: np.random.Generator,
) -> int:
    if not _internal.get("initialized", False):
        initialize(num_actions, rng)

    # --- Example: epsilon-greedy using the app's sample-mean estimates ---
    if rng.random() < EPSILON:
        # Explore
        return int(rng.integers(0, num_actions))
    else:
        # Exploit
        max_q = np.max(q_estimates)
        candidates = np.where(q_estimates == max_q)[0]
        return int(rng.choice(candidates))
