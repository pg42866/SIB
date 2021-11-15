from .Model import Model
from si.util.util import euclidean, accuracy_score
import numpy as np


class KNN(Model):
    def __init__(self, num_neighbors, classification=True):  
        super().__init__()
        self.k = num_neighbors
        self.classification = classification

    def fit(self, dataset):
        self.dataset = dataset
        self.is_fited = True

    def get_neighbors(self, x):  
        # de treino
        dist = euclidean(x, self.dataset.X)  
        indx_sort = np.argsort(dist)  
        return indx_sort[:self.k]  

    def predict(self, x):
        """
        :param x: array de teste
        :return: predicted labels
        """
        neighbors = self.get_neighbors(x)  
        values = self.dataset.Y[neighbors].tolist()  
        if self.classification:
            prediction = max(set(values), key=values.count)
        else:
            prediction = sum(values) / len(values)
        return prediction

    def cost(self):
        pred = np.ma.apply_along_axis(self.predict, axis=0, arr=self.dataset.X.transpose())
        return accuracy_score(pred, self.dataset.Y)