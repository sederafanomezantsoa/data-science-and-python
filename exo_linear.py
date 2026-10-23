import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.linear_model import SGDRegressor

# creation du dataset

np.random.seed(0) # like random_state=42 in sklearn, affect all future random
x,y = make_regression(n_samples=1000,n_features=1,noise=10)
# plt.scatter(x,y)
# plt.show()
model = SGDRegressor(max_iter=1000,eta0 = 0.001)
model.fit(x,y)

print(model.score(x,y))
plt.scatter(x,y)
plt.plot(x,model.predict(x),c='red',lw=3)
plt.show()