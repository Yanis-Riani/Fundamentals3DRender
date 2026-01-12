""" Class used for left/right edge traversal during filling (scanline).
"""


class Edge():
    y_top: int
    x: int
    num: int
    den: int
    inc: int

    def __init__(self, y_top: int = 0, x: int = 0, num: int = 0, den: int = 0, inc: int = 0) -> None:
        self.y_top = y_top
        self.x = x
        self.num = num
        self.den = den
        self.inc = inc

    def update(self) -> None:
        if self.den == 0:
            return
        self.inc += self.num
        Q = self.inc // self.den
        self.x += Q
        self.inc -= Q * self.den
