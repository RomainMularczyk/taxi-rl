import numpy as np
from lib.models.Action import Action
from lib.data.q_table import QTable


class QLearning:
    """
    Provide util functions to compute Q-Learning steps.

    Attributes
    ----------
    gamme: float, default=1.0
        The discount factor.
    initial_seed: int
        Initial seed to reset the environment in the same starting state
        for each new episode.
    cutoff_score: int
        If the cutoff score is reached, stop the episode.
    """

    def __init__(
        self,
        initial_seed: int,
        cutoff_score: int,
        observation_space: int,
        action_space: int,
        gamma: float = 1.0,
    ):
        self.initial_seed = initial_seed
        self.cutoff_score = cutoff_score
        self.gamma: float = gamma
        self.qtable = QTable(observation_space, action_space)

    @classmethod
    def train(cls):
        while True:
            # 1. Do step
            # 2. Get reward
            # 3. Have I win / Reached cutoff score ?
            # 4. Update QTable => Bellman equations (QTable(a, s) => retrieve value)
            pass

    @staticmethod
    def expected_value(n: int, p: float) -> float:
        """
        Compute the expected value.

        Parameters
        ----------
        n: int
            Number of outcomes.
        p: float
            Probability of each outcome.

        Returns
        -------
        float
            The expected value.
        """
        e = np.array([i * p for i in range(n + 1)])
        return e.sum()

    def q_function(self, state_index: int, action_index: int):
        """
        Compute the expected return after taking an action (a), starting
        from a given state (s).

        Parameters
        ----------
        state_index: int
            The current state index.
        action_index: int
            The current action index.

        Returns
        -------
        float
            The expected value of the cumulated rewards.
        """
        num_actions = [action.value for action in list(Action)]
        return self.expected_value(
            self.qtable[state_index, action_index],
            num_actions
        )

    @classmethod
    def bellman_equations(cls):
        q[state, action] = q[state, action] + learning_rate_a * (
                    reward + discount_factor_g * np.max(q[new_state, :]) - q[state, action]
                )
