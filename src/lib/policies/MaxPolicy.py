import numpy as np
from typing import List
from gymnasium.wrappers.time_limit import TimeLimit as GymnasiumGameEnvironment
from lib.models.Action import Action, ActionWithReward
from lib.policies.Policy import Policy


class MaxPolicy(Policy):
    """"""

    def __init__(
        self,
        env: GymnasiumGameEnvironment,
        seed: int | None = None
    ) -> None:
        super().__init__(env=env, seed=seed)
        self.type = Policy.Type.DETERMINISTIC

    def possible_actions(self) -> List[Action]:
        """
        Return all the next available actions following the policy.

        Returns
        -------
        List[Action]
            A list of actions.
        """
        return [self.next_action().action]

    def next_action(self) -> ActionWithReward | ValueError:  # type: ignore
        """
        Returns the next action maximizing the immediate reward.

        Returns
        -------
        ActionWithReward
            The optimal next action to take.

        Raises
        ------
        ValueError
            If no optimal action could be found.
        """
        rewards_and_prob = [
            self.take_action(action.value) for action in list(Action)
        ]
        max_action = None
        max_reward = -np.inf
        for el in rewards_and_prob:
            if el.reward > max_reward:
                max_reward = el.reward
                max_action = el.action
        if max_action:
            return ActionWithReward(
                action=max_action,
                reward=max_reward
            )
        else:
            raise ValueError(
                "No optimal action could be found."
            )
