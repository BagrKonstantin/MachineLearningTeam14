import numpy as np
from numpy import ndarray
from random import random, choice

from Entities.Bee import Bee
from Entities.Flower import Flower
from Entities.Object import Object
from Entities.Pollen import Pollen

P_dist = 0.5
P_pickup = 0.5


def normalize(x):
    norm1 = np.linalg.norm(x)
    return x / norm1


def normalized_euclidean_distance(x: ndarray, y: ndarray) -> float:
    return np.linalg.norm(normalize(x) - normalize(y))


def compatibility(obj1: ndarray, obj2: ndarray) -> float:
    return 1 - normalized_euclidean_distance(obj1, obj2)


class Cell:
    def __init__(self):
        self.objects: list[Object] = list()

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

    def put_pollen(self, pollen: Pollen):
        self.objects.append(Pollen(pollen.vec, self.growth_of_pollen(pollen), pollen.n))

    # main functions

    def center(self) -> np.ndarray:
        return 1 / self.size() * np.sum([i.vec for i in self.objects], axis=0)

    def growth_of_pollen(self, obj: Object) -> float:
        return compatibility(obj.vec, self.center())

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
        min_flower = self.flower_with_min_growth()
        pollen: Pollen = bee.pollen

        if min_flower.growth < pollen.growth:
            bee.pick_up(min_flower)
            return True
        if self.growth_of_pollen(pollen) > pollen.growth and not self.is_pollen_present(pollen):
            self.put_pollen(pollen)
        return False

    def pick_up_pollen(self, bee: Bee) -> bool:
        size = self.size()
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
