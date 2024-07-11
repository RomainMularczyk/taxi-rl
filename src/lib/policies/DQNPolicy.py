import torch
from gymnasium.wrappers.time_limit import TimeLimit as GymnasiumGameEnvironment
from lib.data.neural_network import Model
from lib.data.memory import Memory
from lib.models.Action import Action
from lib.policies.Policy import Policy


class DQNPolicy(Policy):

    def __init__(
        self,
        game_env: GymnasiumGameEnvironment,
        policy_network: Model,
        seed: int | None = None
    ) -> None:
        super().__init__(game_env=game_env, seed=seed)
        self.seed = seed
        self.policy_network = policy_network

    def reset_hyperparameters(self, reset_env: bool = False) -> None:
        """
        Resets the environment and the hyperparameters of the policy.

        Parameters
        ----------
        reset_env: bool, default=False
            Decide if the environment should be also reset.
        """
        if reset_env:
            self.game_env.env.reset(seed=self.seed)

    def next_action(self):
        with torch.no_grad():
            state = Memory.one_hot(
                self.game_env.state,
                self.game_env.env.observation_space.n  # type: ignore
            )
            out = self.policy_network(state)
        action = Action(int(torch.argmax(out)))
        return self.take_action(action)
