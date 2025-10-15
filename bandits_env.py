import numpy as np
from dataclasses import dataclass
from typing import Literal

EnvKind = Literal["gaussian", "dice"]

_STANDARD_DICE = (4, 6, 8, 10, 12, 20)

@dataclass
class GaussianSpec:
    means: np.ndarray  
    variances: np.ndarray

@dataclass
class DiceSpec:
    sides: np.ndarray

class BanditEnvironment:
    """
    Currently, there are two modes:

    1) gaussian distributions:
        rewards ~ Normal(mean_i, variance_i)
        You can randomize (means, variances) inside UI-provided limits.

    2) dice distributions:
        rewards ~ Uniform{1,2,...,n_i}, where n_i is a standard die size.

    API defenitions:
      - kind: 'gaussian' | 'dice'
      - reset_gaussian_random(mean_min, mean_max, var_min, var_max)
      - set_dice_sides(sides_per_arm)
      - set_kind(kind, num_arms)
      - pull(arm_index)
      - get_true_params()
    """

    def __init__(self, num_arms: int, rng: np.random.Generator | None = None):
        self.rng = rng or np.random.default_rng()
        self.num_arms = int(num_arms)
        self.kind: EnvKind = "gaussian"
        # Default Gaussian setting
        self.gaussian = GaussianSpec(
            means=np.zeros(self.num_arms, dtype=float),
            variances=np.ones(self.num_arms, dtype=float),
        )
        # Default Dice setting
        self.dice = DiceSpec(
            sides=np.full(self.num_arms, 6, dtype=int)
        )

    # Mode management ----------
    def set_kind(self, kind: EnvKind, num_arms: int | None = None):
        if num_arms is not None and int(num_arms) != self.num_arms:
            self.num_arms = int(num_arms)
            # Resize specs
            self.gaussian = GaussianSpec(
                means=np.zeros(self.num_arms, dtype=float),
                variances=np.ones(self.num_arms, dtype=float),
            )
            self.dice = DiceSpec(
                sides=np.full(self.num_arms, 6, dtype=int)
            )
        self.kind = kind

    def reset_gaussian_random(self, mean_min: float, mean_max: float,
                              var_min: float, var_max: float):
        mean_lo, mean_hi = sorted((float(mean_min), float(mean_max)))
        var_lo, var_hi   = sorted((float(var_min), float(var_max)))
        var_lo = max(var_lo, 1e-6)  # keep sane
        self.gaussian.means = self.rng.uniform(mean_lo, mean_hi, size=self.num_arms)
        self.gaussian.variances = self.rng.uniform(var_lo, var_hi, size=self.num_arms)

    def set_dice_sides(self, sides_per_arm: list[int] | np.ndarray):
        sides = np.asarray(sides_per_arm, dtype=int)
        if sides.shape != (self.num_arms,):
            raise ValueError(f"sides length must be {self.num_arms}")
        for n in sides:
            if int(n) not in _STANDARD_DICE:
                raise ValueError(f"Invalid die size {n}; allowed: {_STANDARD_DICE}")
        self.dice.sides = sides.copy()

    # Acting ----------
    def pull(self, arm_index: int) -> float:
        if not (0 <= arm_index < self.num_arms):
            raise IndexError("arm_index out of range")
        if self.kind == "gaussian":
            m = self.gaussian.means[arm_index]
            v = self.gaussian.variances[arm_index]
            std = float(np.sqrt(v))
            return float(self.rng.normal(loc=m, scale=std))
        else:  # dice
            n = int(self.dice.sides[arm_index])
            # Uniform integer in 1 to n
            return float(self.rng.integers(1, n + 1))

    # Introspection ----------
    def get_true_params(self) -> dict:
        if self.kind == "gaussian":
            return {
                "kind": "gaussian",
                "means": self.gaussian.means.copy(),
                "variances": self.gaussian.variances.copy(),
            }
        else:
            return {
                "kind": "dice",
                "sides": self.dice.sides.copy(),
            }
