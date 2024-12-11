# create a custom exception for if no valid bounds are found

class NoValidBounds(Exception):
    def __init__(self, message="No valid bounds found."):
        self.message = message
        super().__init__(self.message)