import yfinance as yf
import pandas as pd
import numpy as np
import math, datetime
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use('ggplot')

df = yf.download("GOOGL", start="2000-01-01", end="2016-12-31", auto_adjust=False)
df.columns = ['Open', 'High', 'Low', 'Close', 'Adj. Close', 'Volume']

df['HCL_PCT'] = (df['High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Open']) / df['Open'] * 100.0
df = df[['Adj. Close', 'HCL_PCT', 'PCT_change', 'Volume']]

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)

forecast_out = int(math.ceil(0.01 * len(df)))
df['label'] = df[forecast_col].shift(-forecast_out)

X = np.array(df.drop(['label'], axis=1))
X = preprocessing.scale(X)

x_lately = X[-forecast_out:]
X = X[:-forecast_out]

df.dropna(inplace=True)
y = np.array(df['label'])

print(X.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = LinearRegression(n_jobs=-1)
clf.fit(x_train, y_train)

# ✅ Save
with open('linearregression.pickle', 'wb') as f:
    pickle.dump(clf, f)

# ✅ Load (same filename)
with open('linearregression.pickle', 'rb') as f:
    clf = pickle.load(f)

accuracy = clf.score(x_test, y_test)
print("Accuracy:", accuracy)

forecast_set = clf.predict(x_lately)
print("Forecasted prices:", forecast_set)

df['Forecast'] = np.nan
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend()
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()