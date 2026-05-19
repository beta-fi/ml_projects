from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd
import pickle

import matplotlib.pyplot as plt
mnist = fetch_openml('mnist_784', as_frame=False)
X, y = mnist.data, mnist.target
X = X.reshape(70000,784)
y=y.astype(int)
X=X/255

x_train,x_test,y_train,y_test=train_test_split(X,y ,test_size=0.2,random_state=42)
x_train = x_train[:10000]
y_train = y_train[:10000]
x_test= x_test[:2000]
y_test= y_test[:2000]


model=SVC(kernel='rbf')
model.fit(x_train,y_train)
import os

if os.path.exists('mnist_svm.pkl'):
    with open('mnist_svm.pkl', 'rb') as f:
        model = pickle.load(f)
else:
    model = SVC(kernel='rbf')
    model.fit(x_train, y_train)
    with open('mnist_svm.pkl', 'wb') as f:
        pickle.dump(model, f)
model.score(x_test,y_test)
y_pred=model.predict(x_test)
wrong = np.where(y_test!=y_pred)
#testing stuff like how many it got wrong ,accuracy,which number its confusing ,how accurate is every number prediction
print(wrong)
print(model.score(x_test,y_test))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test,y_pred))
for i in wrong[0][:10]:

    image=x_test[i].reshape(28, 28)
    plt.imshow(image)
    plt.title(f"True: {y_test[i]}, Predicted: {y_pred[i]}")
    plt.show()

