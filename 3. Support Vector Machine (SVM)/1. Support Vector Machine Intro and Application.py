import numpy as np
import pandas as pd
from sklearn import model_selection, svm

df = pd.read_csv('../2. K Nearest Neighbour/breast-cancer-wisconsin.data.txt')
df.replace('?', -99999, inplace=True) # -99999 = outlier
df.drop(['id'], 1, inplace=True)

X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y, test_size=0.2) #20%

clf = svm.SVC()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)

print(accuracy)

example_measures = np.array([[4, 2, 1, 1, 1, 2, 3, 2, 1],
                             [4, 2, 1, 2, 2, 2, 3, 2, 1]]) #Both don't exist in document

example_measures = example_measures.reshape(len(example_measures),-1)

prediction = clf.predict(example_measures)
print(prediction)
