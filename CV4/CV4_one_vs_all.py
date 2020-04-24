from sklearn import svm as svm_factory
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
import typing
from sklearn.metrics import f1_score, plot_confusion_matrix
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

        for clazz in self.classes_:
            _y = np.copy(y)
            np.place(_y, _y!=clazz,self.a_class)
            _svm = svm_factory.SVC(kernel='poly')
            _svm.fit(X, _y)
            self._svms[clazz] = _svm

        return self

    def predict(self, X):
        """
        return first class that is not -1

        :param X:
        :return:
        """
        X = check_array(X)
        _cy = np.ones(np.shape(X)[0])*-1

        for k, v in self._svms.items():
            _y = v.predict(X)
            np.copyto(_cy, _y, where=_y!=self.a_class)

        return _cy

svm = SVM_OneVsMany()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
svm = svm.fit(X_train,y_train)
y_pred = svm.predict(X_test)


f1_score_svm = f1_score(y_test, y_pred,  average='micro')
print("one vs all SVM f1 score:"+str(f1_score_svm))


gnb = GaussianNB()
gnb.fit(X_train,y_train)
y_pred = gnb.predict(X_test)

f1_score_gnb = f1_score(y_test, y_pred,  average='micro')
print("naive bayes f1 score:"+str(f1_score_gnb))

plot_confusion_matrix(svm, X_test, y_test, cmap=plt.cm.Blues)
plt.savefig('figs/one_v_all/data'+str(dataset)+'_'+str(suffix)+'_confusion_50_50.png')
plt.clf()

plot_confusion_matrix(svm, X_test, y_test, cmap=plt.cm.Blues)
plt.savefig('figs/naive_bayes/data'+str(dataset)+'_'+str(suffix)+'_confusion_50_50.png')
plt.clf()


