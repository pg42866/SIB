from .Model import Model
import numpy as np
from ..util.util import sigmoide
from ..util.util import add_intersect

class LogisticRegression(Model):
    def __init__(self,epochs = 1000, lr=0.1):
        super(LogisticRegression, self).__init__()
        self.theta = None
        self.epochs = epochs
        self.lr = lr

    def fit(self,dataset):
        X, y = dataset.getXy()
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        self.X = X
        self.y = y
        self.train_gd()
        self.is_fitted = True

    def train_gd(self):
        n = self.X.shape[1]
        self.history ={}
        self.theta = np.zeros(n)
        for epoch in range(self.epochs):
            Z = np.dot(self.X, self.theta)
            h = sigmoide(Z)
            gradient = np.dot(self.X.T, (h-self.y)) / self.y.size
            self.theta -= self.lr * gradient
            self.history[epoch] = [self.theta[:], self.cost()]

    def predict(self, X):
        assert self.is_fitted, 'model must be fitted before predicting'
        _x = np.hstack(([1],X))
        p = sigmoide(np.dot(self.theta, _x))
        if p <=0.5:
            return 0
        else:
            return 1

    def cost(self, X=None, y=None, theta=None):
        X = add_intersect(X) if X is not None else self.X
        y = y if y is not None else self.y
        theta = theta if theta is not None else self.theta

        y_pred = np.dot(X, theta)
        h = sigmoide(y_pred)
        cost = (y * np.log(h) + (1-y) * np.log(1-h))
        res = -np.sum(cost) / X.shape[0]
        return res

class LogisticRegressionReg(LogisticRegression):
    def __init__(self,epochs = 1000, lr=0.1,lbd = 1):
        super(LogisticRegressionReg, self).__init__(epochs=epochs, lr=lr)
        self.lbd = lbd

    def train_gd(self):
        m = self.X.shape[0]
        n = self.X.shape[1]
        self.history ={}
        self.theta = np.zeros(n)
        lbds = np.full(m, self.lbd)
        lbds[0] = 0
        for epoch in range(self.epochs):
            Z = np.dot(self.X, self.theta)
            h = sigmoide(Z)
            grad = np.dot(self.X.T, (h-self.y)) / self.y.size
            grad[1:] = grad[1:] + (self.lbd/m) * self.theta[1:]
            self.theta -= self.lr * grad
            self.history[epoch] = [self.theta[:], self.cost()]

    def predict(self, X):
        assert self.is_fitted, 'model must be fitted before predicting'
        _x = np.hstack(([1],X))
        p = sigmoide(np.dot(self.theta, _x))
        if p <=0.5:
            return 0
        else:
            return 1

    def cost(self, X=None,y=None,theta=None):
        X = add_intersect(X) if X is not None else self.X
        y = y if y is not None else self.y
        theta = theta if theta is not None else self.theta

        m = X.shape[0]
        h = sigmoide(np.dot(X, theta))
        cost = (y * np.log(h) + (1-y) * np.log(1-h))
        reg = np.dot(theta[1:],theta[1:]) * self.lbd / (2*m)
        res = (-np.sum(cost)/m)+reg
        return res