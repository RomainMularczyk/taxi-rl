from typing import Union
from lib.policies.LegalSamplePolicy import LegalSamplePolicy
from lib.policies.RandomSamplePolicy import RandomSamplePolicy
from lib.policies.MonteCarloPolicy import MonteCarloPolicy
from lib.policies.GreedyPolicy import GreedyPolicy
from lib.policies.EpsilonGreedyPolicy import EpsilonGreedyPolicy


Policies = Union[
    EpsilonGreedyPolicy,
    GreedyPolicy,
    LegalSamplePolicy,
    RandomSamplePolicy,
    MonteCarloPolicy
]
