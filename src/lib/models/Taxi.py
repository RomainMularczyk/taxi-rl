class Taxi():
    """
    Representation of the taxi location.
    """

    @staticmethod
    def location(state: int):
        """
        Convert the current state to a given location on the
        environment grid.
        """
        row = (state // 100) % 5
        col = (state // 20) % 5
        return row, col
