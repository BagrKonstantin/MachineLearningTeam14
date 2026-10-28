from sklearn import datasets
from Jungle import Jungle

if __name__ == '__main__':
    data = datasets.load_iris().data
    jungle = Jungle(data)
    jungle.run(3)