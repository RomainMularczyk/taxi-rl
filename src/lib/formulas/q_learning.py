import numpy as np
from typing import List
from tqdm import tqdm
from lib.models.Action import Action, ActionWithReward
from lib.models.GameStatus import GameStatus
from lib.models.Metrics import EpisodeMetrics, StepResult
from lib.models.Policies import Policies
from lib.models.Metrics import Result
from lib.data.q_table import QTable
from lib.policies.GreedyPolicy import GreedyPolicy
from lib.policies.EpsilonGreedyPolicy import EpsilonGreedyPolicy
from lib.policies.MonteCarloPolicy import MonteCarloPolicy
from lib.policies.Policy import Policy


class QLearning:
    """
    Provide utility functions to compute Q-Learning steps.

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
        stable_env: bool = True,
        advantage: bool = True,
        max_steps: int = 100,
        gamma: float = 1.0,
        lr: float = 1.0
    ) -> None:
        self.max_steps = max_steps
        self.gamma = gamma
        self.lr = lr
        self.stable_env = stable_env
        self.advantage = advantage
        if data is None:
            self.data = QTable(observation_space, action_space)
        else:
            self.data = data
        self.policy = policy

    def do_step(self) -> StepResult:
        """
        Execute a single step of the game.

        Returns
        -------
        StepResult
            The status of the game (terminated, truncated, running)
            and the immediate reward for taking the current step.
        """
        current_state = self.policy.game_env.state
        next = self.q_star(current_state)
        # Update Q-Table // `self.policy.game_env.state` is updated,
        # corresponding to `next_state`
        self.data.values[current_state, next.action.value] = next.bellman
        return StepResult(
            action=next.action,
            game_status=next.game_status,
            immediate_reward=next.reward
        )

    def do_episode(self) -> EpisodeMetrics:
        """
        Execute an entire episode. The episode stops either because :
        - The maximum number of steps is reached (200)
        - The game exited because the winning condition was reached
        - The game exited because a losing condition was reached

        Returns
        -------
        EpisodeMetrics
            The number of steps and the cumulative reward.
        """
        self.policy.reset_hyperparameters(reset_env=True)
        steps = 0
        cumulative_reward = 0
        while steps < self.max_steps:
            step_result = self.do_step()
            cumulative_reward += step_result.immediate_reward
            steps += 1
            if step_result.game_status == GameStatus.TERMINATED:
                break
        return EpisodeMetrics(
            steps=steps,
            cumulative_reward=cumulative_reward
        )

    def train(self, num_episodes: int) -> List[EpisodeMetrics]:
        """
        Train the model for a given number of episodes.

        Parameters
        ----------
        num_episodes: int
            The number of episodes.

        Returns
        -------
        List[EpisodeMetrics]
            A list of episode metrics.
        """
        metrics = []
        for _ in tqdm(range(num_episodes)):
            metrics.append(self.do_episode())
        return metrics

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
        Compute the expected value for each possible action at a given state.

        Returns
        -------
            The expected value for each possible action at a given state.
        """
        return np.array([  # type: ignore
            next.probability * self.data[  # type: ignore
                self.policy.game_env.state, next.action.value
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
    ) -> Result:
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
        Result
            The action taken by the optimal policy, associated with both :
            - The immediate reward
            - The result of computing the Bellman equation
            - The game environment status (terminated, truncated, running)

        Raises
        ------
        GameExitException
            If the game exits in an unexpected manner.
        """
        # Take the new action and update state
        if type(self.policy) is GreedyPolicy \
           or type(self.policy) is EpsilonGreedyPolicy \
           or type(self.policy) is MonteCarloPolicy:
            current = self.policy.next_action(self.v_star(action=True))  # type: ignore
        else:
            current = self.policy.next_action()  # type: ignore

        if advantage:
            return Result(
                action=current.action,
                reward=current.reward,
                bellman=self.lr * (
                    current.reward + self.gamma * self.v_star()  # type: ignore
                    - self.q(current_state, current.action)
                ) + self.q(current_state, current.action),
                game_status=current.game_status
            )
        else:
            return Result(
                action=current.action,
                reward=current.reward,
                bellman=self.lr * (
                    current.reward + self.gamma * self.v_star()  # type: ignore
                ) + self.q(current_state, current.action),
                game_status=current.game_status
            )
