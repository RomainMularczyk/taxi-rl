from abc import ABCMeta, abstractmethod
from enum import Enum
from typing import List
from gymnasium.wrappers.time_limit import TimeLimit as GymnasiumGameEnvironment
from lib.environment.environment import GameEnvironment
from lib.models.Action import Action, ActionWithReward


class Policy(metaclass=ABCMeta):
    """
    Abstract representation of a policy.
    """

    class Type(Enum):
        PROBABILISTIC = "P"
        DETERMINISTIC = "D"

    def __init__(
        self,
        env: GymnasiumGameEnvironment,
        seed: int | None
    ):
        self.env = GameEnvironment(env=env, seed=seed)

    @abstractmethod
    def next_action(self) -> ActionWithReward:
        pass

    @abstractmethod
    def possible_actions(self) -> List[Action]:
        pass

    def take_action(self, action: Action | int) -> ActionWithReward:
        """
        Take a given action.

        Parameters
        ----------
        action: Action | int
            The action taken.

        Returns
        -------
        float
            The reward for taking the action.

        Raises
        ------
        TypeError
            If the action is neither an integer or an Action.
        """
        if type(action) is int:
            new_state, reward, _, _, _ = self.env.env.step(
                action
            )
            self.env.state = new_state
            return ActionWithReward(
                action=Action(action),
                reward=float(reward)
            )
        elif type(action) is Action:
            new_state, reward, _, _, _ = self.env.env.step(
                action.value
            )
            self.env.state = new_state
            return ActionWithReward(
                action=action,
                reward=float(reward)
            )
        else:
            raise TypeError(
                "The action should either be an integer or an Action."
            )
