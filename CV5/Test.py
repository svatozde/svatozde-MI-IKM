from CV5.EnsembleKnn import EnsembleKnn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


dataset = ''
dataset_suffix = 'tst'

X = np.genfromtxt('../CV3/csv/data'+str(dataset)+'_x_'+str(dataset_suffix)+'.csv', delimiter=',', skip_header=1)
y = np.genfromtxt('../CV3/csv/data'+str(dataset)+'_y_'+str(dataset_suffix)+'.csv', delimiter=',', skip_header=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

ensemble = EnsembleKnn(n_estimators=20, bag_size=0.85)
K = 5
ensemble.fit(X_train,y_train)

y_pred = ensemble.predict(X_test)

score = f1_score(y_test, y_pred, average='micro')
print(score)

