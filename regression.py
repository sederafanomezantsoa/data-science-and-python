import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split


data = pd.read_csv("data.csv")
print(data.head())
print(data.columns)

x=data[["distance_km","weight_kg","traffic_level",'rain_mm']]
y=data["delivery_time_min"]

X_train, X_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

reg = linear_model.LinearRegression()
reg.fit(X_train,y_train)
print(reg.coef_)

print("we are going to test out prediction : \n")

y_pred = reg.predict(X_test)


results = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred
})

results["Error"] = results["Actual"] - results["Predicted"]

print(results)