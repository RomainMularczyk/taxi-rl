from lib.data.monte_carlo_tree import MonteCarloTree
from lib.data.q_table import QTable


class MonteCarloTreeSearchFormula:
    """
    Provide util functions to compute MonteCarlo steps.

    Attributes
    ----------
    """
    
    def __init__(
        self,
        data: MonteCarloTree | QTable | None = None
    ) -> None:
        self.data = data

    def train(self):
        pass