import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, RepeatedKFold, RepeatedStratifiedKFold
from sklearn.metrics import f1_score, make_scorer, plot_confusion_matrix
from sklearn.metrics import confusion_matrix


dataset = '3'

X_train = np.genfromtxt('csv/data'+str(dataset)+'_x_tst.csv', delimiter=',', skip_header=1)
Y_train = np.genfromtxt('csv/data'+str(dataset)+'_y_tst.csv', delimiter=',', skip_header=1)

X_test = np.genfromtxt('csv/data'+str(dataset)+'_x_tr.csv', delimiter=',', skip_header=1)
Y_test = np.genfromtxt('csv/data'+str(dataset)+'_y_tr.csv', delimiter=',', skip_header=1)

nbrs = KNeighborsClassifier()

ks = [i for i in range(1, 10)]

samples = 100

metrics = [
    'euclidean',
    'manhattan',
    'chebyshev',
    'minkowski',
    'wminkowski',
    'cosine'
]

hyper_params = {
    'n_neighbors': ks,
    'metric': metrics
}

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5)

search = GridSearchCV(
    nbrs,
    hyper_params,
    scoring=make_scorer(f1_score,average='micro'),
    verbose=3,
    return_train_score=True,
    n_jobs=-1,
    cv=cv
)
search.fit(X_train, Y_train)
Y_pred = search.predict(X_test)

f1 = f1_score(Y_test, Y_pred, average='micro')
print("*** F1 score: " + str(f1))

plot_confusion_matrix(search, X_test, Y_test,cmap=plt.cm.Blues)
plt.savefig('figs/data'+str(dataset)+'_confusion.png')
plt.clf()

series = {m: [0 for _ in ks] for m in metrics}
times = {m: [0 for _ in ks] for m in metrics}

params = search.cv_results_['params']
for index, par in enumerate(params):
    series[par['metric']][par['n_neighbors'] - 1] = search.cv_results_['mean_test_score'][index]
    times[par['metric']][par['n_neighbors'] - 1] = search.cv_results_['mean_score_time'][index]

for metric, series in series.items():
    plt.plot(series, label=metric);

plt.legend()
plt.xlabel('k')
plt.ylabel('F1 score')
plt.savefig('figs/data'+str(dataset)+'_knn_f1_score-tst.png')
plt.clf()

for metric, series in times.items():
    plt.plot(series, label=metric)

plt.legend()
plt.xlabel('k')
plt.ylabel('time')
# plt.show()
plt.savefig('figs/data'+str(dataset)+'_knn_times-tst.png')

plt.clf()
