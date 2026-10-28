from numpy import ndarray

class Object:
    def __init__(self, vec: ndarray, growth: float, n: int, art: bool = False):
        self.vec: ndarray = vec
        self.growth: float = growth
        self.n: int = n
        self.art: bool = art