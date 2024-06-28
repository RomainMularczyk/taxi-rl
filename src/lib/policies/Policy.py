from gymnasium.wrappers.time_limit import TimeLimit as GymnasiumGameEnvironment
from lib.environment.environment import GameEnvironment
from lib.models.Action import Action


class Policy:

    def __init__(
        self,
        env: GymnasiumGameEnvironment,
        seed: int | None
    ) -> None:
        self.env = GameEnvironment(env, seed=seed)

    def take_action(self, action: Action) -> float:
        """
        Take a given action.

        Parameters
        ----------
        action: Action
            The action taken.

        Returns
        -------
        float
            The reward for taking the action.
        """
        _, reward, _, _, _ = self.env.env.step(
            action.value
        )
        return float(reward)
