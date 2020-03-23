import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import pairwise_distances

from sklearn.metrics import f1_score, make_scorer, plot_confusion_matrix


def knn(D, Y, k, num_samles = -1):
    """
    :param D: Distance matrix
    :param Y: Labels
    :param k: k
    :param num_samles: how many labels shoudl be randomly selected and checked
    :return: recalculated labels for each element
    """
    sample_size = Y.size
    indexes = np.arange(sample_size)
    if num_samles > 0:
        sample_size = num_samles;
        indexes = np.random.choice(indexes,size=min(sample_size,indexes.size), replace=False);

    ret = np.zeros(indexes.size)
    for i, index in np.ndenumerate(indexes):
        idx = np.argpartition(D[index], k + 1)[:k + 1]
        idx = idx[idx != index]  # remove self reference
        unique, counts = np.unique(np.take(Y,idx), return_counts=True) # count class occurences in closest neighbors
        ret[i] = unique[np.argmax(counts)]

    return ret, indexes


dataset_x = 'data2_x_tr.csv'
dataset_y = 'data2_y_tr.csv'

X = np.genfromtxt('csv/' + str(dataset_x), delimiter=',', skip_header=1)
Y = np.genfromtxt('csv/' + str(dataset_y), delimiter=',', skip_header=1)

D = pairwise_distances(X, metric='manhattan')

results = []
for k in range(1, 20):
    Y_pred, Y_indexes = knn(D, Y, k, num_samles=2500)
    f1 = f1_score(np.take(Y,Y_indexes).astype(int), Y_pred.astype(int), average='micro')
    results.append(f1)
    print("*** F1 for k:" + str(k) + " score: " + str(f1))

plt.plot(results);

plt.legend()
plt.xlabel('k')
plt.ylabel('F1 score')
plt.savefig('figs/my_knn_rassigment_'+dataset_x+'.png')

print(D)
