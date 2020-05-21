import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from itertools import product
from sklearn.decomposition import PCA

from CV5.EnsembleKnn import EnsembleKnn

dataset = ''
dataset_suffix = 'tr'

X = np.genfromtxt('../CV3/csv/data'+str(dataset)+'_x_'+str(dataset_suffix)+'.csv', delimiter=',', skip_header=1)
y = np.genfromtxt('../CV3/csv/data'+str(dataset)+'_y_'+str(dataset_suffix)+'.csv', delimiter=',', skip_header=1)

#X,_,y,_ = train_test_split(X, y, test_size=0.95, random_state=42) #reduce data for visualization

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

def knnToLabel(ensemble: EnsembleKnn, index: int) -> str:
    return 'k=' + str(ensemble.estimators[index].n_neighbors)+ ' f1=' +  str(ensemble.f_scores[index])

# Training classifiers

eclf = EnsembleKnn(n_estimators=15, bag_size=0.35, k=7)

eclf.fit(X_train, y_train)

pca = PCA(n_components=2)

pca.fit(X)
Xviz = pca.transform(X)

# Plotting decision regions
step = 200

x_min, x_max = Xviz[:, 0].min() - 1, Xviz[:, 0].max() + 1
y_min, y_max = Xviz[:, 1].min() - 1, Xviz[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, (x_max - x_min)/step),
                     np.arange(y_min, y_max, (y_max - y_min)/step))

random_indices = np.random.choice(Xviz.shape[0], min(500,Xviz.shape[0]), replace=False)

f, axarr = plt.subplots(1, 2, sharex='col', sharey='row', figsize=(10, 8))

ensembled_f1 = f1_score(y_test, eclf.predict(X_test), average='micro')

for idx, clf, tt in zip(product([0], [0, 1]),
                        [*eclf.estimators, eclf],
                        [knnToLabel(eclf,i) for i,_ in enumerate(eclf.estimators)] + ["ensembled f1=" + str(ensembled_f1)]):
    XX = np.c_[xx.ravel(), yy.ravel()]
    XX = pca.inverse_transform(XX)

    Z = clf.predict(XX)
    Z = Z.reshape((step,step))
    axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.4)
    axarr[idx[0], idx[1]].scatter(Xviz[random_indices, 0], Xviz[random_indices, 1], c=y[random_indices],
                                  s=20, edgecolor='k')
    axarr[idx[0], idx[1]].set_title(tt)

plt.show()
print()