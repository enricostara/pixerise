from pixerise.canvas import Canvas
from pixerise.viewport import ViewPort


class RayTracer:

    def __init__(self, canvas: Canvas, viewport: ViewPort, scene: object):
        self._canvas = canvas
        self._viewport = viewport
        self._scene = scene

    def render(self, origin: (float, float, float)):
        self._canvas.draw_unchecked_point(0, 0, (0, 255, 0))

    def _render(self, origin: (float, float, float), direction: (float, float, float)):
        pass
