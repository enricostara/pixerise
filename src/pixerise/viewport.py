from pixerise.canvas import Canvas


class ViewPort:

    def __init__(self, w: float, h: float, d: float, canvas: Canvas):
        self.w = w
        self.h = h
        self.d = d
        self._canvas = canvas

    def canvas_to_viewport(self, x, y):
        return x * self.w / self._canvas.size[0], y * self.h / self._canvas.size[1], self.d
