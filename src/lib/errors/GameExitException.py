class GameExitException(Exception):
    """
    Exception raised if the game exits in an unexepcted manner.
    """

    def __init__(self, msg: str | None = None):
        if msg is None:
            self.msg = "The game exited in an unexpected manner."
