from sklearn import svm as svm_factory
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
import typing
from sklearn.metrics import f1_score, make_scorer, plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB

dataset = '3'
suffix = 'tr'

X = np.genfromtxt('../CV3/csv/data' + str(dataset) + '_x_'+str(suffix)+'.csv', delimiter=',', skip_header=1)
y = np.genfromtxt('../CV3/csv/data' + str(dataset) + '_y_'+str(suffix)+'.csv', delimiter=',', skip_header=1)


class SVM_OneVsMany(BaseEstimator, ClassifierMixin):

    def __init__(self,  kernel='rbf'):
        self.classes_ = None
        self._svms = {}
        self.a_class = -1
        self._kernel = kernel

    def fit(self, X, y):
        y = y.astype(int)
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)

        for aa_class in range(-1,-100, -1):
            if aa_class not in self.classes_:
                self.a_class = aa_class
                break

        for i,clazz in enumerate(self.classes_):
            for clazz2 in self.classes_[i+1:]:
                indexes = np.append(np.argwhere(y==clazz), np.argwhere(y==clazz2))
                _svm = svm_factory.SVC(kernel='poly')
                X_copy = X[indexes]
                Y_copy = y[indexes]
                _svm.fit(X_copy, Y_copy)
                if clazz not in self._svms:
                    self._svms[clazz] = {}
                self._svms[clazz][clazz2] = _svm

        return self

    def predict(self, X):
        """
        return first class that is not -1

        :param X:
        :return:
        """
        X = check_array(X)
        _cy = None

        for _, svmMap in self._svms.items():
            for _, v in svmMap.items():
                _y = v.predict(X)
                if _cy is None:
                    _cy = np.copy(_y)
                else:
                    _cy = np.row_stack((_cy,_y))

        ret_len = _cy.shape[len(_cy.shape)-1]

        ret = np.zeros(ret_len);

        for i in range(0, ret_len):
            column = _cy[:,i]
            ret[i] = np.bincount(column).argmax()

        return ret


svm = SVM_OneVsMany()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=42)
svm = svm.fit(X_train,y_train)
y_pred = svm.predict(X_test)

f1_score_svm = f1_score(y_test, y_pred,  average='micro')
print('One v One SVM f1 score:' + str(f1_score_svm))


gnb = GaussianNB()
gnb.fit(X_train,y_train)
y_pred = gnb.predict(X_test)

f1_score_gnb = f1_score(y_test, y_pred,  average='micro')
print("naive bayes f1 score:"+str(f1_score_gnb))

plot_confusion_matrix(svm, X_test, y_test, cmap=plt.cm.Blues)
plt.savefig('figs/one_v_one/data'+str(dataset)+'_'+str(suffix)+'_confusion_50_50.png')
plt.clf()

plot_confusion_matrix(gnb, X_test, y_test, cmap=plt.cm.Blues)
plt.savefig('figs/naive_bayes/data'+str(dataset)+'_'+str(suffix)+'_confusion_50_50.png')
plt.clf()



