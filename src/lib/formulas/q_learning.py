from numpy import ndarray
from lib.models.Policies import Policies
from lib.models.Action import Action
from lib.data.q_table import QTable


class QLearning:
    """
    Provide util functions to compute Q-Learning steps.

    Attributes
    ----------
    gamma: float, default=1.0
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
        policy: Policies,
        gamma: float = 1.0,
    ):
        self.initial_seed = initial_seed
        self.cutoff_score = cutoff_score
        self.gamma: float = gamma
        self.data = QTable(observation_space, action_space)
        self.policy = policy

    @classmethod
    def train(cls):
        while True:
            # 1. Do step
            # 2. Get reward
            # 3. Have I win / Reached cutoff score ?
            # 4. Update QTable => Bellman equations (QTable(a, s) => retrieve value)
            pass

    def expected_value(self, n: ndarray) -> float:
        """
        Compute the expected value.

        Parameters
        ----------
        n: int
            Number of outcomes.

        Returns
        -------
        float
            The expected value.
        """
        return (n * self.policy.p).sum()

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
        for action in list(Action):
            self.policy.take_action(action)
        self.data[state_index, action_index]
        return self.expected_value(len(num_actions))

    def bellman_equations(self):
        return self.expected_value(
            self.gamma * self.expected_value(self.q_function(state_index, action_index))
        )
        q[state, action] = q[state, action] + learning_rate_a * (
                    reward + discount_factor_g * np.max(q[new_state, :]) - q[state, action]
                )
