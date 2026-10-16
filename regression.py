import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt



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

print("\n\n*****************Descent_gradient********\n\n")

def descent_gradient(x,y):
    m_curr=b_curr=0
    iterations = 1000
    n=len(x)
    learning_rate = 0.001

    MSE_history = []
    m_history = []
    b_history = []
    for i in range(iterations):
        y_predict = m_curr*x +b_curr
        MSE = (1/n)*sum([v**2 for v in (y-y_predict)])
        MSE_history.append(MSE)
        md = (-2/n)*sum(x*(y-y_predict))
        bd = (-2/n)*sum(y-y_predict)
        #md <-- d/dm((1/n)(sum²(yi-(mxi+b)))

        m_curr = m_curr - learning_rate*md
        b_curr = b_curr - learning_rate*bd
        m_history.append(m_curr)
        b_history.append(b_curr)
        print(f"m = {m_curr}") 
        print(f"b = {b_curr}") 
        print(f"iterations = {i}") 
        print(f"MSE = {MSE}")   
    return m_curr,b_curr,MSE_history,m_history,b_history
x = np.array([1,2,3,4,5,6,7,8,9])
y = np.array([5,8,4,6,7,12,6,5,13])

m,b,m_hist, b_hist, mse_hist = descent_gradient(x, y)

# --- VISUALIZATION 1: Evolution of Regression Lines ---
plt.figure(figsize=(8, 5))

# 1. Plot the original data points
plt.scatter(x, y, color='black', zorder=5, label='Data points')

# 2. Plot the regression line at every iteration
for i in range(len(m_hist)):
    y_pred = m_hist[i] * x + b_hist[i]
    # Use alpha (transparency) so overlapping lines don't just look like a solid block
    plt.plot(x, y_pred, color='green', alpha=0.15, linewidth=1)

# 3. Highlight the final line
y_final = m_hist[-1] * x + b_hist[-1]
plt.plot(x, y_final, color='red', linewidth=2, label='Final model (iteration 100)')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Evolution of Regression Lines during Gradient Descent')
plt.legend()
plt.grid(True)
plt.show()