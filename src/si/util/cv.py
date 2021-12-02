from src.si.util.util import train_test_split, add_intersect
import numpy as np
import itertools


class CrossValidationScore:
    def __init__(self, model, dataset, **kwargs):
        self.model = model
        self.dataset = dataset
        self.cv = kwargs.get('cv', 3)
        self.split = kwargs.get('split', 0.8)
        self.train_scores=None
        self.test_scores = None
        self.ds = None

    def run(self):
        train_scores = []
        test_scores = []
        ds = []
        for _ in range(self.cv):
            train, test = train_test_split(self.dataset,self.split)
            ds.append((train,test))
            self.model.fit(train)
            train_scores.append(self.model.cost())
            test_scores.append(self.model.cost(test.X,test.Y))
        self.train_scores = train_scores
        self.test_scores = test_scores
        self.ds = ds
        return train_scores,test_scores

    def toDataframe(self):
        import pandas as pd
        assert self.train_scores and self.test_scores, 'need to run model'
        return pd.DataFrame({'train Scores':self.train_scores, 'Test Scores': self.test_scores})