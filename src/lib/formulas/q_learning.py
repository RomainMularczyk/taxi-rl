import numpy as np
from typing import List, Tuple
from tqdm import tqdm
from lib.models.Policies import Policies
from lib.models.Action import Action, ActionWithReward
from lib.data.q_table import QTable
from lib.policies.Policy import Policy


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
        observation_space: int,
        action_space: int,
        policy: Policies,
        data: QTable | None = None,
        cutoff_score: int = -100_000,
        max_steps: int = 100,
        gamma: float = 1.0,
    ):
        self.cutoff_score = cutoff_score
        self.max_steps = max_steps
        self.gamma: float = gamma
        if data is None:
            self.data = QTable(observation_space, action_space)
        else:
            self.data = data
        self.policy = policy

    def train(self):
        steps = 0
        cumulative_reward = 0
        with tqdm(total=self.max_steps):
            while cumulative_reward > self.cutoff_score and steps < self.max_steps:
                action, reward, result = self.q_star()
                self.data[self.policy.env.state, action.value] = result
                cumulative_reward += reward
                steps += 1

    @staticmethod
    def expected_value(n: np.ndarray, p: np.ndarray) -> float:
        """
        Compute the expected value.

        Parameters
        ----------
        n: ndarray
            Value of each outcome.
        p: ndarray
            Probabily of each outcome.

        Returns
        -------
        float
            The expected value.
        """
        return (n * p).sum()

    def j(self) -> float:
        """
        Compute the average expected value for each possible action at
        a given state.

        Returns
        -------
            The expected value for each possible action at a given state.
        """
        return np.array([  # type: ignore
            next.probability * self.data[  # type: ignore
                self.policy.env.state, next.action.value
            ]
            for next in self.rewards()
        ]).sum()

    def rewards(self) -> List[ActionWithReward]:
        """
        Collect the rewards for the possible actions provided by the policy.

        Returns
        -------
        List[ActionWithReward]
            A list of all actions associated with their reward and probability.
        """
        if self.policy.type is Policy.Type.PROBABILISTIC:
            actions_reward = []
            actions_probability = self.policy.actions_probability()  # type: ignore
            for action in self.policy.possible_actions():  # type: ignore
                action_with_reward = self.policy.take_action(action)
                action_with_reward.probability = actions_probability\
                    .model_dump()[action.name]  # type: ignore
                actions_reward.append(action_with_reward)
            return actions_reward
        else:
            raise TypeError(
                "Rewards can only be cumulated on probabilistic policies."
            )

    def v(self, action: Action):
        self.j()

    def v_star(self):
        pass

    def q(self, action: Action) -> float:
        """
        Compute the result for the Q-function for a given action.

        Parameters
        ----------
        action: Action
            A given action.

        Returns
        -------
        float
            The reward for taking the provided action.
        """
        next = self.policy.take_action(action)
        return next.reward + self.gamma * self.j()

    def q_star(
            self,
            mask: np.ndarray | None = None
    ) -> Tuple[Action, float, float]:
        """
        Compute the result for the Q-function following the optimal action.

        Parameters
        ----------
        mask: np.ndarray, default=None
            The possible actions into which to sample.

        Returns
        -------
        Tuple[Action, float, float]
            A tuple associating the action taken by the optimal policy,
            the direct reward and the Q-value for taking that action.
        """
        next = self.policy.next_action(mask)  # type: ignore
        return (
            next.action,
            next.reward,
            float(next.reward + self.gamma * self.j())
        )
