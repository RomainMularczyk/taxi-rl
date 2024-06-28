from typing import Union
from lib.policies.LegalSamplePolicy import LegalSamplePolicy
from lib.policies.RandomSamplePolicy import RandomSamplePolicy
from lib.policies.MonteCarloPolicy import MonteCarloPolicy


Policies = Union[
    LegalSamplePolicy,
    RandomSamplePolicy,
    MonteCarloPolicy
]
