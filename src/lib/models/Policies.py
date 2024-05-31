from typing import Union
from lib.policies.SamplePolicy import SamplePolicy
from lib.policies.MonteCarloPolicy import MonteCarloPolicy


Policies = Union[SamplePolicy, MonteCarloPolicy]
