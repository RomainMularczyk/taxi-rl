from gymnasium.wrappers.time_limit import TimeLimit as GymnasiumGameEnvironment


class MaxPolicy:
    """"""
    p: float = 1.0

    @staticmethod
    def next_action(
        env: GymnasiumGameEnvironment,
        human_readable: bool = False
    ):
        pass
