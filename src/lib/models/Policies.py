from typing import Union
from lib.policies.LegalSamplePolicy import LegalSamplePolicy
from lib.policies.RandomSamplePolicy import RandomSamplePolicy
from lib.policies.MonteCarloPolicy import MonteCarloPolicy
from lib.policies.MaxPolicy import MaxPolicy


Policies = Union[
    MaxPolicy,
    LegalSamplePolicy,
    RandomSamplePolicy,
    MonteCarloPolicy
]
