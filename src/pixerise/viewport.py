from pixerise.canvas import Canvas


class ViewPort:

    def __init__(self, size: (float, float), plane_distance: float, canvas: Canvas):
        self._width = size[0]
        self._height = size[1]
        self._plane_distance = plane_distance
        self._canvas = canvas

    def viewport_to_canvas(self, x, y) -> (float, float):
        return x *  self._canvas.width / self._width, y *  self._canvas.height / self._height
