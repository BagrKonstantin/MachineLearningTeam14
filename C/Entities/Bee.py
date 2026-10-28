from Entities.Object import Object
import Entities.Pollen as Pollen


class Bee:
    def __init__(self, x, y):
        self.pollen: Pollen.Pollen | None = None
        self.x = x
        self.y = y

    def pick_up(self, obj: Object):
        self.pollen = Pollen.get_pollen(obj)

    def clear(self):
        self.pollen = None

    def has_pollen(self) -> bool:
        return self.pollen is not None
