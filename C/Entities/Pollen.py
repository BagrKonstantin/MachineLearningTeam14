from Entities.Object import Object
from numpy import ndarray


class Pollen(Object):
    def __init__(self, vec: ndarray, growth: float, n: int):
        super().__init__(vec, growth, n)


def get_pollen(obj: Object) -> Pollen:
    return Pollen(obj.vec, obj.growth, obj.n)
