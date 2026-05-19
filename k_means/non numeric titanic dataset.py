import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.cluster import KMeans, MeanShift
from sklearn import preprocessing
import pandas as pd
import numpy as np

df = pd.read_excel(r"C:\Users\Gaurav Kumar\Downloads\titanic.xls")
original_df = pd.DataFrame.copy(df)
df.drop(['body','name'], axis=1, inplace=True)

def handle_non_data(df):
    columns = df.columns.values
    for column in columns:
        test_digit_vals = {}
        def convert_to_int(val):
            return test_digit_vals[val]
        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in test_digit_vals:
                    test_digit_vals[unique] = x
                    x += 1
            df[column] = list(map(convert_to_int, df[column]))
    return df

df = handle_non_data(df)
df.fillna(0, inplace=True)

x = np.array(df.drop(['survived'], axis=1).astype(float))
x = preprocessing.scale(x)
y = np.array(df['survived'])

model = MeanShift()
model.fit(x)
labels = model.labels_
cluster_centers = model.cluster_centers_

original_df['cluster_group'] = np.nan
for i in range(len(x)):
    original_df['cluster_group'].iloc[i] = labels[i]

survival_rates = {}
for i in range(len(model.cluster_centers_)):
    temp_df = original_df[original_df['cluster_group'] == float(i)]
    survival_cluster = temp_df[temp_df['survived'] == 1]
    survival_rates[i] = len(survival_cluster)/len(temp_df)

print(survival_rates)