
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from  typing import List
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
import random

class EnsembleKnn(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=5, k=1, bag_size=0.75):
        self.estimators: List[KNeighborsClassifier] = []
        self.f_scores: List[float] = []
        self._n_estimators: int = n_estimators
        self._bag_size: float = bag_size
        self._k = k

    def fit(self, X, y):
        for i in range(0,self._n_estimators):
            est, f1 = self._create_classifier(X,y)
            self.estimators.append(est)
            self.f_scores.append(f1)

    def _create_classifier(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=self._bag_size, random_state=random.randint(0,100))
        ret = KNeighborsClassifier(n_neighbors=min(self._k, len(X_train)))
        ret.fit(X_train, y_train)
        y_pred = ret.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='micro')
        return ret, f1

    def predict(self, X):
        y_res = None
        for c in self.estimators:
           y_pred = c.predict(X)
           if y_res is None:
               y_res = np.copy(y_pred)
           else:
               y_res = np.row_stack((y_res, y_pred))

        ret_len = y_res.shape[len(y_res.shape) - 1]
        ret = np.zeros(ret_len)

        for i in range(0, ret_len):
            column = y_res[:, i].astype('int64')
            ret[i] = np.bincount(column, weights=self.f_scores).argmax()

        return ret
