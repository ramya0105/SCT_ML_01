import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = pd.read_csv('house_prices.csv')

X = data[['SquareFootage', 'Bedrooms', 'Bathrooms']]
y = data['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(X_test['SquareFootage'], y_test, color='blue', label='Actual Prices')
plt.plot(X_test['SquareFootage'], model.predict(X_test[['SquareFootage', 'Bedrooms', 'Bathrooms']]), color='red', label='Fitted Line')
plt.xlabel('Square Footage')
plt.ylabel('Price')
plt.title('Square Footage vs Price')
plt.legend()

plt.subplot(1, 3, 2)
plt.scatter(X_test['Bedrooms'], y_test, color='blue', label='Actual Prices')
bedroom_range = pd.DataFrame({'SquareFootage': X_test['SquareFootage'].mean(), 'Bedrooms': range(int(X['Bedrooms'].min()), int(X['Bedrooms'].max() + 1)), 'Bathrooms': X_test['Bathrooms'].mean()})
price_pred_bedrooms = model.predict(bedroom_range)
plt.plot(bedroom_range['Bedrooms'], price_pred_bedrooms, color='red', label='Fitted Line')
plt.xlabel('Bedrooms')
plt.ylabel('Price')
plt.title('Bedrooms vs Price')
plt.legend()

plt.subplot(1, 3, 3)
plt.scatter(X_test['Bathrooms'], y_test, color='blue', label='Actual Prices')
bathroom_range = pd.DataFrame({'SquareFootage': X_test['SquareFootage'].mean(), 'Bedrooms': X_test['Bedrooms'].mean(), 'Bathrooms': range(int(X['Bathrooms'].min()), int(X['Bathrooms'].max() + 1))})
price_pred_bathrooms = model.predict(bathroom_range)
plt.plot(bathroom_range['Bathrooms'], price_pred_bathrooms, color='red', label='Fitted Line')
plt.xlabel('Bathrooms')
plt.ylabel('Price')
plt.title('Bathrooms vs Price')
plt.legend()

plt.tight_layout()
plt.show()
