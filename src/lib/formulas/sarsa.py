import numpy as np
from collections import namedtuple
from typing import List, Tuple
from tqdm import tqdm
from lib.errors.GameExitException import GameExitException
from lib.models.Action import Action, ActionWithReward
from lib.models.GameStatus import GameStatus
from lib.models.Policies import Policies
from lib.data.q_table import QTable
from lib.policies.GreedyPolicy import GreedyPolicy
from lib.policies.EpsilonGreedyPolicy import EpsilonGreedyPolicy


class Sarsa:
    """
    Provide utility functions to compute Sarsa steps.
    """

    def __init__(
        self,
        observation_space: int,
        action_space: int,
        policy: Policies,
        data: QTable | None = None,
        max_steps: int = 100,
        gamma: float = 1.0,
        lr: float = 1.0
    ):
        self.max_steps = max_steps
        self.gamma = gamma
        self.lr = lr
        if data is None:
            self.data = QTable(observation_space, action_space)
        else:
            self.data = data
        self.policy = policy

    def train(self):
        steps = 0
        cumulative_reward = 0
        with tqdm(total=self.max_steps):
            while steps < self.max_steps:  # TODO: add cutoff_score
                action, reward, result = self.q_star()
                self.data[self.policy.game_env.state, action.value] = result

                cumulative_reward += reward
                steps += 1

    def reward(self) -> ActionWithReward:
        pass

    def v(self) -> np.ndarray:
        """
        Row of the Q-Table corresponding to the current state.

        Returns
        -------
        np.ndarray
            Row of the Q-Table corresponding to the current state.
        """
        return self.data.values[self.policy.game_env.state]

    def v_star(self, action: bool = False) -> float | Action:
        """
        Maximum value of the Q-Table row corresponding to the current state.

        Parameters
        ----------
        action: bool, default=False
            Output the best action according to the Q-values instead of the
            Q-value itself.

        Returns
        -------
        float | Action
            Maximum value of the Q-Table row corresponding to the current
            state. If 'action' is True, returns the best action instead.
        """
        if action:
            return Action(np.argmax(self.v()))
        return self.v().max()

    def q(self, state: int, action: Action) -> float:
        """
        Compute the result for the Q-function for a given action
        knowing the state.

        Parameters
        ----------
        state: int
            A given state.
        action: Action
            A given action.

        Returns
        -------
        float
            The reward for taking the provided action.
        """
        return self.data.values[state, action.value]

    def q_star(
        self,
        current_state: int,
        advantage: bool = True,
        mask: np.ndarray | None = None
    ) -> Tuple[Action, float, float, GameStatus]:
        """
        Compute the result for the Q-function following the optimal action.

        Parameters
        ----------
        current_state: int
            The current state.
        mask: np.ndarray, default=None
            The possible actions into which to sample.
        advantage: bool, default=True
            Defines if the method should compute the difference with the
            previous Q-values.

        Returns
        -------
        Tuple[Action, float, float]
            The action taken by the optimal policy, associated with both :
            - The immediate reward
            - The result of computing the Bellman equation
            - The game environment status (terminated, truncated, running)

        Raises
        ------
        GameExitException
            If the game exits in an unexpected manner.
        """
        # Create tuple template
        Result = namedtuple(
            "result",
            ("action", "reward", "bellman", "game_status")
        )
        # Take the new action and update state
        if type(self.policy) is GreedyPolicy \
           or type(self.policy) is EpsilonGreedyPolicy:
            current = self.policy.next_action(self.v_star(action=True), mask)  # type: ignore
        else:
            current = self.policy.next_action(mask)  # type: ignore

        if type(current) is ActionWithReward:
            if advantage:
                return Result(
                    current.action,
                    current.reward,
                    self.lr * (
                        current.reward + self.gamma * self.v_star()  # type: ignore
                        - self.q(current_state, current.action)
                    ) + self.q(current_state, current.action),
                    GameStatus.RUNNING
                )
            else:
                return (
                    current.action,
                    current.reward,
                    self.lr * (
                        current.reward + self.gamma * self.v_star()  # type: ignore
                    ) + self.q(current_state, current.action)
                )
        else:
            if current == GameStatus.TERMINATED:
                return Result(
                    current.action,
                    current.reward,
                    0.0,
                    GameStatus.TERMINATED
                )
            elif current == GameStatus.TRUNCATED:
                return Result(
                    current.action,
                    current.reward,
                    0.0,
                    GameStatus.TRUNCATED
                )
            else:
                raise GameExitException

