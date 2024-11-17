from pixerise.canvas import Canvas


class ViewPort:

    def __init__(self, width: float, height: float, distance: float, canvas: Canvas):
        self.width = width
        self.height = height
        self.distance = distance
        self._canvas = canvas

    def canvas_to_viewport(self, x, y):
        return x * self.width / self._canvas.width, y * self.height / self._canvas.height, self.distance
