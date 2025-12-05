from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# load dataset

data = pd.read_csv(
    './advertising.csv', usecols=['TV', 'Sales'])
X = data[['TV']]
y = data['Sales']

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# create and train the model
model = LinearRegression()
model.fit(X_train, y_train)
# make predictions
y_pred = model.predict(X_test)
# print the model coefficients
print(f'Intercept: {model.intercept_}')
print(f'Coefficient: {model.coef_[0]}')
print(f'Model score on training set: {model.score(X_train, y_train)}')

# evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')
# make a prediction for a new value
new_tv_budget = [[150]]  # $150
predicted_sales = model.predict(new_tv_budget)
print(
    f'Predicted sales for a TV advertising budget of $150: ${predicted_sales[0]:.2f}')

# graph the regression line
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regression Line')
plt.xlabel('TV Advertising Budget ($)')
plt.ylabel('Sales ($)')
plt.title('TV Advertising Budget vs Sales with Regression Line')
plt.legend()
plt.show()

# Save the model using joblib
