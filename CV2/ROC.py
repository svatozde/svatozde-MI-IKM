import pandas as pd
import matplotlib
data_1 = pd.read_csv('C:\skola\MI-IKM\CV2\data\csv\dataset1_y_tr.csv')
data_2 = pd.read_csv('C:\skola\MI-IKM\CV2\data\csv\dataset1_y_tr.csv')
data_1['label'] = data_2


fig = matplotlib.figure()
data_1.hist(by='label')
fig.show()

print()
