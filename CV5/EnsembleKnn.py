
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from  typing import List


class EnsembleKnn(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.classifiers: List[KNeighborsClassifier] = []
        self.f_scores = []

    def fit(self, X, y,K):
        classifiers = [ KNeighborsClassifier(n_neighbors=k) for k in range(1,K) ]
        for c in classifiers:
            c.fit(X,y)
        return self

    def predict(self, X):
        y_res = None
        for c in self.classifiers:
           y_pred = c.predict(X)
           if y_res is None:
               y_res = np.copy(y_pred)
           else:
               y_res = np.row_stack((y_res, y_pred))

        ret_len = y_res.shape[len(y_res.shape) - 1]
        ret = np.zeros(ret_len);

        for i in range(0, ret_len):
            column = y_res[:, i]
            ret[i] = np.bincount(column).argmax()

        return ret
