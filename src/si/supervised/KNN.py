from .Model import Model
from src.si.util.util import euclidean
from src.si.util.metrics import accuracy_score
import numpy as np


class KNN(Model):
    def __init__(self, number_neighboors, classification=True):
        super(KNN).__init__()
        self.number_neighboors = number_neighboors
        self.classification = classification

    def fit(self, dataset):
        self.dataset = dataset
        self.is_fitted = True

    def get_neighboors(self, x):

        """
        Calcula as distâncias entre cada ponto de teste
        em relação a todos os pontos do dataset de treino
        """

        distance = euclidean(x,
                             self.dataset.X)
        idxs_sort = np.argsort(distance)
        return idxs_sort[:self.number_neighboors]

    def predict(self, x):

        """
        :param x: array de teste
        :return: predicted labels
        """

        assert self.is_fitted, 'Model must be fot before prediction'
        viz = self.get_neighboors(x)
        values = self.dataset.Y[viz].tolist()
        if self.classification:
            prediction = max(set(values),
                             key=values.count)
        else:
            prediction = sum(values) / len(values)
        return prediction

    def cost(self):
        y_pred = np.ma.apply_along_axis(self.predict, axis=0,
                                        arr=self.dataset.X.T)
        return accuracy_score(self.dataset.Y, y_pred)