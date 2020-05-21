import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import  load_files
from sklearn.datasets.base import Bunch
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from scipy import sparse as sp

from sklearn.feature_extraction.text import  TfidfVectorizer,TfidfTransformer

data: Bunch = load_files('../SEM/data/',encoding='utf-8',decode_error='replace')
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data.data)
y = data.target
pca = TruncatedSVD(n_components=2)
pca.fit(X)
Xviz = pca.transform(X)
random_indices = np.random.choice(Xviz.shape[0], min(5000,Xviz.shape[0]), replace=False)
plt.figure(figsize=(10, 8))
plt.scatter(Xviz[random_indices, 0], Xviz[random_indices, 1], c=y[random_indices],s=20, edgecolor='k')
plt.show()

print()