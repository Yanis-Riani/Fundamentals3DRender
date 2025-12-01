""" classe utilisee pour le parcours gauche ou droite des aretes pour le remplissage
"""


class Arete():
    yhaut: int
    x: int
    num: int
    den: int
    inc: int

    def __init__(self, yhaut1: int = 0, x1: int = 0, num1: int = 0, den1: int = 0, inc1: int = 0) -> None:
        self.yhaut = yhaut1
        self.x = x1
        self.num = num1
        self.den = den1
        self.inc = inc1

    def maj(self) -> None:
        if self.den == 0:
            return
        self.inc += self.num
        Q = self.inc // self.den
        self.x += Q
        self.inc -= Q * self.den