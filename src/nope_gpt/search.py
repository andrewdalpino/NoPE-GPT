from math import exp

from dataclasses import dataclass

from torch import Tensor


@dataclass
class Candidate:
    """A candidate sequence used in beam search."""

    cumulative_log_probability: float
    tokens: Tensor
    length_penalty: float

    @property
    def cumulative_probability(self) -> float:
        return exp(self.cumulative_log_probability)

    @property
    def priority(self) -> float:
        return self.cumulative_log_probability / self.length_norm

    @property
    def length_norm(self) -> float:
        return len(self.tokens) ** self.length_penalty
