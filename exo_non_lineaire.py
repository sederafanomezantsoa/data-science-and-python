import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import SGDRegressor

np.random.seed(0)
x,y = make_regression(n_samples=500, n_features=1,noise=15)
y = y**2 # avadika le data fa tsy lineaire tsony

poly_features = PolynomialFeatures(degree=2, include_bias=False)
x=poly_features.fit_transform(x) #asina polynome ao amin ilay x

plt.scatter(x[:,0],y)
print(x.shape)
# print(x[:,0])
# print(x[:,1])
# plt.show()

# trainning of the model

model = SGDRegressor(max_iter=1000, eta0=0.001)
model.fit(x,y)
print(model.score(x,y))
plt.scatter(x[:,0],y,marker='.')
plt.scatter(x[:,0],model.predict(x),c="red",marker='+')
plt.show()