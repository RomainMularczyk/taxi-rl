class GameAlreadyWonException(Exception):
    """
    Exception raised if the we want to take a `step()`
    on the environment when the game is already finished
    meaning the passenger has been successfully dropped off.
    """

    def __init__(self, msg: str | None = None):
        if msg is None:
            self.msg = "Game finished ! The passenger has already been successfully dropped off."
