from typing import Union
from lib.policies.DynamicSamplePolicy import DynamicSamplePolicy
from lib.policies.StaticSamplePolicy import StaticSamplePolicy
from lib.policies.MonteCarloPolicy import MonteCarloPolicy


Policies = Union[
    DynamicSamplePolicy,
    StaticSamplePolicy,
    MonteCarloPolicy
]
