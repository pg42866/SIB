import numpy as np
from src.si.util.scale import StandardScaler

class PCA:
    def __init__(self, num_components=2, using="svd"):
        self.numcomps = num_components
        self.alg = using

#    def fit(self, dataset):  # objeto Dataset
#        pass

    def transform(self, dataset):  # objeto Dataset
        x_scaled = StandardScaler.fit_transform(dataset)  # standardização dos dados usando o StandardScaler

        matriz_cov = np.cov(x_scaled, rowvar=False)  #

        self.eigen_values, self.eigen_vectors = np.linalg.eigh(matriz_cov)

        # sort the eigenvalues in descending order
        self.sorted_index = np.argsort(self.eigen_values)[::-1]  # np.argsort returns an array of indices of the same shape.
        self.sorted_eigenvalue = self.eigen_values[self.sorted_index]
        # similarly sort the eigenvectors
        sorted_eigenvectors = self.eigen_vectors[:, self.sorted_index]

        # select the first n eigenvectors, n is desired dimension of our final reduced data.
        # you can select any number of components.
        eigenvector_subset = sorted_eigenvectors[:, 0:self.numcomps]

        x_reduced = np.dot(eigenvector_subset.transpose(), x_scaled.transpose()).transpose()
        return x_reduced

    def explained_variance(self, dataset):
        self.sorted_eigenvalue_sub = self.sorted_eigenvalue[0:self.numcomps]
        return np.sum(self.sorted_eigenvalue_sub), self.sorted_eigenvalue_sub

    def fit_transform(self, dataset):
        data_reduced = self.transform(dataset)
        explain, eigvalues = self.explained_variance(dataset)
        return data_reduced, explain, eigvalues