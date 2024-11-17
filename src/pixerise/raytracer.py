from pixerise.canvas import Canvas
from pixerise.viewport import ViewPort


class RayTracer:

    def __init__(self, canvas: Canvas):
        self._canvas = canvas
        self._viewport = ViewPort(1, 1, 1, canvas)
