from xgboost import XGBRegressor





import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv(r"C:\Users\Gaurav Kumar\Downloads\data.csv", parse_dates=["date"])


df = df[df["price"] > 10000]


q99 = df["price"].quantile(0.98)
df = df[df["price"] < q99]

print("Max price after cleaning:", df["price"].max())
print("Rows remaining:", len(df))
df["yr_renovated"] = df.apply(lambda x: x["yr_built"] if x["yr_renovated"] == 0 else x["yr_renovated"], axis=1)
df["price_per_sqft"] = df["sqft_living"] / df["sqft_lot"]


df = df.drop(["date", "street", "statezip", "country"], axis=1)
df=pd.get_dummies(df,columns=["city"])
df = df.astype({col: int for col in df.select_dtypes(bool).columns})
x=df.drop("price",axis=1)
y=df["price"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=XGBRegressor(n_estimators=1000,learning_rate=0.005,random_state=42)
model.fit(x_train,y_train)
model.score(x_test,y_test)
print(model.score(x_test,y_test))
print(x.isnull().sum().sum())  # total nulls in X
print(y.isnull().sum())        # nulls in y
print(x.shape)                 # how many rows and columns

