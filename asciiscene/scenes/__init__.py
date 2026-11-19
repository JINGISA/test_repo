"""Scenes. A scene turns a time `t` into a colour image, one pixel per character cell.

To add one: create scenes/<name>.py with a Scene subclass, then register it in SCENES below.
"""


class Scene:
    """Base class. Subclasses set PARAMS (the tunable defaults) and implement image(t)."""

    PARAMS = {}

    def __init__(self, cols, rows, **overrides):
        unknown = set(overrides) - set(self.PARAMS)
        if unknown:
            raise KeyError(f"unknown parameter(s) {sorted(unknown)}; "
                           f"available: {sorted(self.PARAMS)}")
        self.cols, self.rows = cols, rows
        self.p = {**self.PARAMS, **overrides}
        self.setup()

    def setup(self):
        """Precompute anything that does not change between frames."""

    def image(self, t):
        """Return a float array (rows, cols, 3) with values in 0..1."""
        raise NotImplementedError


from .aurora import Aurora  # noqa: E402  (needs Scene defined above)

SCENES = {"aurora": Aurora}
