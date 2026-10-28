from numpy import ndarray
from math import sqrt
from random import randint

import skfuzzy as fuzz
import matplotlib.pyplot as plt
import numpy as np

from Entities.Flower import Flower
from Cell import Cell
from Entities.Bee import Bee


def plot(data, u_orig, cntr):
    fig2, ax2 = plt.subplots()
    for j in range(len(cntr)):
        ax2.plot(data[0, u_orig.argmax(axis=0) == j],
                 data[1, u_orig.argmax(axis=0) == j], 'o',
                 label='cluster ' + str(j + 1))
    for pt in cntr:
        ax2.plot(pt[0], pt[1], 'rx')
    ax2.legend(loc="lower right")
    plt.show()


class Jungle:
    def __init__(self, data: ndarray[ndarray]):
        self.n: int = len(data)
        self.m: int = int(sqrt(4 * self.n))
        self.grid: list[list[Cell]] = [[Cell() for _ in range(self.m)] for _ in range(self.m)]
        self.bees: list[Bee] = list()
        self.init_cells(data)
        self.init_bees()

    def init_cells(self, data: ndarray[ndarray]):
        self.n = 0
        for i, vec in enumerate(data):
            x, y = self.get_rand_point()
            while not self.grid[x][y].is_empty():
                x, y = self.get_rand_point()
            self.grid[x][y].put_flower(Flower(vec, 0, i))
        self.n += len(data)

    def init_bees(self):
        for i in range(self.n // 3):
            self.bees.append(Bee(*self.get_rand_point()))

    def get_rand_point(self):
        return randint(0, self.m - 1), randint(0, self.m - 1)

    def natural_selection(self):
        growth_of_flowers = dict()
        for i in range(self.m):
            for j in range(self.m):
                if self.grid[i][j].is_empty():
                    continue
                for obj in self.grid[i][j].objects:
                    if obj.n not in growth_of_flowers:
                        growth_of_flowers[obj.n] = obj.growth
                    if obj.growth > growth_of_flowers[obj.n]:
                        growth_of_flowers[obj.n] = obj.growth
        for i in range(self.m):
            for j in range(self.m):
                if not self.grid[i][j].is_garden():
                    continue
                t = 0
                cell: Cell = self.grid[i][j]
                while t != cell.size():
                    flower = cell.objects[t]
                    if flower.growth < growth_of_flowers[flower.n]:
                        cell.objects.remove(flower)
                        for obj in cell.objects:
                            obj.growth = cell.growth_of_pollen(obj)
                    else:
                        t += 1

    def fly_bee(self, bee: Bee):
        x = bee.x
        y = bee.y
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == j == 0:
                    continue
                bee.x = (x + i) % self.m
                bee.y = (y + j) % self.m
                if self.grid[bee.x][bee.y].is_empty():
                    continue
                if bee.has_pollen():
                    if self.grid[bee.x][bee.y].pollinate(bee):
                        return
                else:
                    if self.grid[bee.x][bee.y].pick_up_pollen(bee):
                        return

    def count_clusters(self) -> int:
        clusters = 0
        for row in self.grid:
            for cell in row:
                if cell.is_garden():
                    clusters += 1
        return clusters

    def find_centroids(self):
        clusters = self.count_clusters()
        data = []
        for i in range(self.m):
            for j in range(self.m):
                for obj in self.grid[i][j].objects:
                    if not obj.art:
                        data.append(obj.vec)

        data = np.array(data)
        data = data.T
        cntr, u_orig, _, _, _, _, _ = fuzz.cluster.cmeans(
            data, clusters, 2, error=0.005, maxiter=1000)

        plot(data, u_orig, cntr)

        return cntr

    def merge(self) -> int:
        cntr = self.find_centroids()
        for i, vec in enumerate(cntr):
            x = randint(0, self.m - 1)
            y = randint(0, self.m - 1)
            while not self.grid[x][y].is_empty():
                x = randint(0, self.m - 1)
                y = randint(0, self.m - 1)
            self.grid[x][y].put_flower(Flower(vec, 0, self.n + i, True))
        return len(cntr)

    def run(self, clusters: int):
        for i in range(100):
            for bee in self.bees:
                self.fly_bee(bee)
            self.natural_selection()
        if self.merge() > clusters:
            self.run(clusters)
