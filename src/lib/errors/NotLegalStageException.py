class NotLegalStageException(Exception):
    """
    Exception raised if the stage provided is not legal.
    """

    def __init__(self, msg: str | None = None):
        if msg is None:
            self.msg = "The provided stage is not legal."
