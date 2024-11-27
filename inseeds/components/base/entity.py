class Entity:
    """Define properties.
    Inherits from I.World as the interface with all necessary variables
    and parameters.
    """

    def __init__(self, model=None, **kwargs):
        """Initialize an instance of World."""
        super().__init__(**kwargs)
        self.model = model
