import numpy as np
from numpy import ndarray
from random import random, choice

from Entities.Bee import Bee
from Entities.Flower import Flower
from Entities.Object import Object
from Entities.Pollen import Pollen

P_dist = 0.5
P_pickup = 0.5


def normalized_euclidean_distance(x: ndarray, y: ndarray, d_max: float = 1.0) -> float:
    if d_max <= 0:
        d_max = 1.0
    return float(np.linalg.norm(x - y) / d_max)


def compatibility(obj1: ndarray, obj2: ndarray, d_max: float = 1.0) -> float:
    return max(0.0, 1.0 - normalized_euclidean_distance(obj1, obj2, d_max))


class Cell:
    def __init__(self, d_max: float = 1.0):
        self.objects: list[Object] = list()
        self.d_max = d_max

    # utils

    def size(self) -> int:
        return len(self.objects)

    def is_pollen_present(self, pollen: Object) -> bool:
        for obj in self.objects:
            if obj.n == pollen.n:
                return True
        return False

    def is_garden(self) -> bool:
        return self.size() >= 2

    def is_empty(self) -> bool:
        return self.size() == 0

    def put_flower(self, flower: Flower):
        self.objects.append(flower)
        self.update_all_growths()

    def put_pollen(self, pollen: Pollen):
        cand_growth = self.growth_of_pollen(pollen)
        self.objects.append(Pollen(pollen.vec, cand_growth, pollen.n))
        self.update_all_growths()

    # main functions

    def center(self) -> np.ndarray:
        if self.is_empty():
            return np.zeros(1)
        return np.mean([i.vec for i in self.objects], axis=0)

    def growth_of_pollen(self, obj: Object) -> float:
        if self.is_empty():
            return 0.0
        return compatibility(obj.vec, self.center(), self.d_max)

    def update_all_growths(self):
        if self.size() <= 1:
            for obj in self.objects:
                obj.growth = 0.0
        else:
            c = self.center()
            for obj in self.objects:
                obj.growth = compatibility(obj.vec, c, self.d_max)

    def flower_with_min_growth(self) -> Object:
        minimum = 1e9
        ind = 0
        for i in range(self.size()):
            cur = self.objects[i].growth
            if cur < minimum:
                minimum = cur
                ind = i
        return self.objects[ind]

    def pollinate(self, bee: Bee) -> bool:
        if self.is_empty() or not bee.has_pollen():
            return False

        min_flower = self.flower_with_min_growth()
        pollen: Pollen = bee.pollen
        cand_growth = self.growth_of_pollen(pollen)

        if min_flower.growth < pollen.growth and self.size() > 1:
            bee.pick_up(min_flower)
            min_flower.vec = pollen.vec
            min_flower.growth = cand_growth
            min_flower.n = pollen.n
            self.update_all_growths()
            return True

        if cand_growth > pollen.growth and not self.is_pollen_present(pollen):
            self.put_pollen(pollen)
            bee.clear()
            return True

        return False

    def pick_up_pollen(self, bee: Bee) -> bool:
        size = self.size()
        if size == 0:
            return False
        if size == 1:
            if random() > P_pickup:
                bee.pick_up(self.objects[0])
                return True
        elif size == 2:
            if random() > P_dist:
                bee.pick_up(choice(self.objects))
                return True
        else:
            bee.pick_up(self.flower_with_min_growth())
            return True
        return False
