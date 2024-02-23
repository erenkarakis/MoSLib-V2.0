import numpy as np # linear algebra

import matplotlib as mpl # ploting
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression  # liner regression model
from sklearn.preprocessing import PolynomialFeatures # polynommial features(extended features)

n = 10 # 11 data points
X = np.array([[0.06171875], [0.0765625], [0.0984375], [0.121875], [0.15078124], [0.18359375], [0.225], [0.275], [0.33984375], [0.42890626], [0.5515625]])
y = np.array([[300.0], [280.0], [260.0], [240.0], [220.0], [200.0], [180.0], [160.0], [140.0], [120.0], [100.0]])

# plotting the dataset
plt.plot(X,y,'b.')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Non-linear Dataset')
plt.grid(True)

poly_features = PolynomialFeatures(degree=2) # decide the maximal degree of the polynomial feature
X_ploy = poly_features.fit_transform(X) # convert the original feature to polynomial feature

# check the extened polynomial features of the first data point
print('original feature:', X[0])
print('polynomial features',X_ploy[0])

lin_reg = LinearRegression()
lin_reg.fit(X_ploy,y)
lin_reg.intercept_, lin_reg.coef_ # check the bais term and feature weights of the trained model

X_new = np.sort(X,axis = 0) # in order to plot the line of the model, we need to sort the the value of x-axis
X_new_ploy = poly_features.fit_transform(X_new) # compute the polynomial features 
print(X_new_ploy)
y_predict = lin_reg.predict([[1,0.065,0.065**2]]) # make predictions using trained Linear Regression model
# y_predict = lin_reg.predict(X_new_ploy)
print(y_predict)
# plot the original dataset and the prediction results
fig,ax = plt.subplots()
ax.plot(X,y,'b.', label = 'Training date samples')
ax.plot(X_new,y_predict,'g-',linewidth=2, label = 'Predictions')
ax.set_xlabel('X')
ax.set_ylabel('y')
ax.legend()
ax.grid(True)
plt.show()