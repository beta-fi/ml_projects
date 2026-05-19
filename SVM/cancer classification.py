from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np
from sklearn import preprocessing, neighbors,svm
from sklearn.model_selection import train_test_split

# Load data
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Define X and y
X = np.array(df.drop(['target'], axis=1))
y = np.array(df['target'])

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)
#preprocessing feautures
scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


clf = svm.SVC()
clf.fit(X_train, y_train)

print("Accuracy:", clf.score(X_test, y_test))


example_measures = np.array([4,5,5,6,1,3,4,2,3,5,1,3,4,5,6,7,8,3,2,3,2,7,8,9,2,4,4,5,6,4])
example_measures = example_measures.reshape(1,-1)
prediction = clf.predict(example_measures)
print("Prediction:", "Benign" if prediction[0] == 1 else "Malignant")