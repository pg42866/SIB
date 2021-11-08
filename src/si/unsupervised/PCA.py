import numpy as np
from src.si.util.scale import StandardScaler
import pandas as pd

class PCA:
    def __init__(self, num_components=2, using="svd"):
        self.numcomps = num_components
        self.alg = using

    def transform(self, dataset):  # objeto Dataset
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(dataset)  # standardização dos dados usando o StandardScaler
        X = x_scaled.X
        XT = X.transpose()
        print(XT)
        self.eigen_vectors, self.eigen_values, vt = np.linalg.svd(XT)
        # sort the eigenvalues in descending order
        self.sorted_index = np.argsort(self.eigen_values)[::-1]  # devolve os indices de acordo com a importância dos valores por ordem crescente
        self.sorted_eigenvalue = self.eigen_values[self.sorted_index]  # ordernar os valores de acordo com os indices anteriores
        # similarly sort the eigenvectors
        sorted_eigenvectors = self.eigen_vectors[:,self.sorted_index]  # devolve apenas as colunas com os indices dos valores

        # select the first n eigenvectors, n is desired dimension of our final reduced data.
        # you can select any number of components.
        eigenvector_subset = sorted_eigenvectors[:, 0:self.numcomps]  # apenas devolve o numero de colunas que queremos
        x_reduced = np.dot(eigenvector_subset.transpose(), XT).transpose()
        return x_reduced

    def variance_explained(self):
        somapercent = np.sum(self.sorted_eigenvalue)
        percentagem = []
        for i in self.sorted_eigenvalue:
            percentagem.append(i/somapercent *100)
        return np.array(percentagem)

    def fit_transform(self, dataset):
        data_reduced = self.transform(dataset)
        percentagem = self.variance_explained()
        return data_reduced, percentagem