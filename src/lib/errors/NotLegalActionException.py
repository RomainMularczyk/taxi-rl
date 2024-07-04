class NotLegalActionException(Exception):
    """
    Exception raised if the action provided is not legal.
    """

    def __init__(self, msg: str | None = None):
        if msg is None:
            self.msg = "The provided action is not legal."
