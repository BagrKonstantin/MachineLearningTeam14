from Entities.Object import Object
from numpy import ndarray


class Flower(Object):
    def __init__(self, vec: ndarray, growth: float, n: int, art: bool = False):
        super().__init__(vec, growth, n, art)
