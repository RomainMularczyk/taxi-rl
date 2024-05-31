from gymnasium.wrappers.time_limit import TimeLimit as GymnasiumGameEnvironment
from lib.models.Action import Action


class StaticSamplePolicy:
    """
    Sample randomly the next action to take from all
    the theoretical actions (including those that are
    not 'legal' moves, i.e. : driving into a wall).

    Attributes
    ----------
    p: float
        The probability of a given action to be taken.
    """
    p: float = 1/6

    @staticmethod
    def next_action(
        env: GymnasiumGameEnvironment,
        human_readable: bool = False
    ):
        """
        Choose the optimal next action.

        Parameters
        ----------
        env: GameEnvironement
            The game environement.
        human_readable: bool, default=False
            Output the action chosen in a human readable format.

        Returns
        -------
        int | str
            The action chosen.
        """
        action = env.action_space.sample()
        if human_readable:
            return Action.to_human_readable(action)
        return action
