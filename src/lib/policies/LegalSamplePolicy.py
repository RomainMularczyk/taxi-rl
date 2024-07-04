import random
import numpy as np
from typing import List
from gymnasium.wrappers.time_limit import TimeLimit as GymnasiumGameEnvironment
from lib.models.Action import Action, ActionProbabilities, ActionWithReward
from lib.policies.Policy import Policy


class LegalSamplePolicy(Policy):
    """
    Sample randomly the next action to take from all
    the available actions in a given state 's' (excluding
    those that are not 'legal' moves, i.e. : driving into
    a wall).

    Attributes
    ----------
    type: Policy.Type
        Defines if the policy is deterministic or probabilistic.
    """

    def __init__(
        self,
        env: GymnasiumGameEnvironment,
        seed: int | None = None
    ) -> None:
        super().__init__(env=env, seed=seed)
        self.type = Policy.Type.PROBABILISTIC

    def actions_probability(self) -> ActionProbabilities:
        """
        Returns the probability associated with each action.

        Returns
        -------
        ActionProbabilities
            The probability associated with each legal action.
        """
        legal_actions_names = [action.name for action in self.legal_actions()]
        num_legal_actions = len(
            self.env.info.action_mask[self.env.info.action_mask == 1]
        )
        action_prob = {
            action: prob for action, prob in zip(
                legal_actions_names, list(
                    np.full(num_legal_actions, 1/num_legal_actions)
                )
            )
        }
        return ActionProbabilities(**action_prob)

    def legal_actions(self) -> List[Action]:
        """
        Returns the legal actions at a given state.

        Returns
        -------
        List[Action]
            A list of legal actions.
        """
        legal_actions = [
            action for action, legal in zip(
                list(Action), self.env.info.action_mask
            ) if legal == 1
        ]
        return legal_actions

    def possible_actions(self) -> List[Action]:
        """
        Return all the next available actions following the policy.

        Returns
        -------
        List[Action]
            A list of actions.
        """
        return self.legal_actions()

    def next_action(  # type: ignore
        self,
        mask: np.ndarray | None,
        probability: float | None = None
    ) -> ActionWithReward:
        """
        Choose the optimal next action.

        Returns
        -------
        Action
            The action chosen.
        """
        if mask is not None:
            action = self.env.env.action_space.sample(mask)
            new_state, reward, _, _, _ = self.env.env.step(action)
            self.env.state = new_state
            return ActionWithReward(
                action=Action(action),
                reward=float(reward),
                probability=probability
            )
        else:
            action = random.choice(self.legal_actions())
            action = self.take_action(action)
            return ActionWithReward(
                action=action.action,
                reward=action.reward,
                probability=float(1/len(self.legal_actions()))
            )
