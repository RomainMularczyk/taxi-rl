import numpy as np
from typing import List
from gymnasium.wrappers.time_limit import TimeLimit as GymnasiumGameEnvironment
from lib.models.Action import Action, ActionProbabilities, ActionWithReward
from lib.policies.Policy import Policy


class RandomSamplePolicy(Policy):
    """
    Sample randomly the next action to take from all
    the theoretical actions (including those that are
    not 'legal' moves, i.e. : driving into a wall).

    Attributes
    ----------
    p: float
        The probability of a given action to be taken.
    env: GameEnvironment
        The game environment.
    reward: float
        The reward collected for taking a given step.
    state: int
        The seed representing a given state of the game environment.
    """

    def __init__(
        self,
        env: GymnasiumGameEnvironment,
        seed: int | None = None
    ):
        super().__init__(env=env, seed=seed)
        self.reward = 0.0
        self.type = Policy.Type.PROBABILISTIC

    @staticmethod
    def actions_probability() -> ActionProbabilities:
        """
        The actions associated with their outcome probability.

        Returns
        -------
        ActionProbabilities
            Action associated with their probabilities.
        """
        action_names = [action.name for action in list(Action)]
        action_prob = {
            action: prob for action, prob in zip(
                action_names, list(np.full(6, 1/6))
            )
        }
        return ActionProbabilities(**action_prob)

    def possible_actions(self) -> List[Action]:
        """
        Return all the next available actions following the policy.

        Returns
        -------
        List[Action]
            A list of actions.
        """
        return list(Action)

    def next_action(  # type: ignore
        self,
        mask: np.ndarray | None = None
    ) -> ActionWithReward:
        """
        Choose the optimal next action.

        Parameters
        ----------
        mask: ndarray | None, default=None
            The possible actions into which to sample.

        Returns
        -------
        ActionWithReward
            The action chosen associated with the reward from taking it.
        """
        if mask is not None:
            action = self.env.env.action_space.sample(mask)
        else:
            action = self.env.env.action_space.sample()

        new_state, reward, _, _, _ = self.env.env.step(action)
        self.env.state = new_state

        return ActionWithReward(
            action=Action(action),
            reward=float(reward),
            probability=float(1/6)
        )
