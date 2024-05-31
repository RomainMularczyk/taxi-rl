import numpy as np


class QLearning:
    """
    Provide util functions to compute Q-Learning steps.

    Attributes
    ----------
    gamme: float, default=1.0
        The discount factor.
    """
    gamma: float = 1.0

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
        e = [i * p for i in range(n + 1)]
        e = np.array(e)
        return e.sum()

    @classmethod
    def cumulative_reward(
        cls,
        rs: float,
        r: float,
    ):
        """
        Compute the cumulative rewards.

        Parmaters
        ---------
        rs: float
            The current accumulated rewards.
        r: float
            The next reward value.

        Returns
        -------
        float
            The accumulated rewards, taking into account the next reward value.
        """
        return rs + (cls.gamma * r)

    @classmethod
    def q_function(cls, rs: float, s, a):
        """
        Compute the expected return after taking an action (a), starting
        from a given state (s).

        Parameters
        ----------
        rs: float
            The cumulated rewards.
        s: float
            The next state.
        a: float
            The next action.

        Returns
        -------
        float
            The expected value of the cumulated rewards.
        """
        return cls.expected_value(cls.cumulative_reward(rs))
