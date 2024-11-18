from pixerise.canvas import Canvas


class ViewPort:

    def __init__(self, size: (float, float), plane_distance: float, canvas: Canvas):
        self._width = size[0]
        self._height = size[1]
        self._plane_distance = plane_distance
        self._canvas = canvas

    def canvas_to_viewport_direction(self, x, y) -> (float, float, float):
        return x * self._width / self._canvas.width, y * self._height / self._canvas.height, self._plane_distance
