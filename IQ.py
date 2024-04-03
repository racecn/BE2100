import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = {
    'Birth Year': [1971, 1971, 1971, 1974, 1976, 1970, 1972, 1975, 1975, 1974, 1973, 1972, 1975, 1977, 1977, 1979, 1978, 1978, 1979, 1981, 1984, 1984, 1987, 1988, 1982, 1983, 1988, 1989, 1983, 1982, 1985, 1986, 1980, 1981, 1981, 1986, 1984, 1983, 1987],
    'IQ Score': [102, 107, 109, 112, 113, 110, 89, 87, 115, 80, 94, 97, 101, 112, 98, 92, 87, 114, 118, 118, 116, 93, 89, 107, 112, 100, 97, 84, 92, 111, 86, 106, 115, 91, 109, 97, 104, 89, 108]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Fit a linear regression model
X = df[['Birth Year']]
y = df['IQ Score']
model = LinearRegression()
model.fit(X, y)

# Make predictions
predictions = model.predict(X)

# Plot the data and the regression line
plt.scatter(X, y, color='blue')
plt.plot(X, predictions, color='red', linewidth=2)
plt.xlabel('Birth Year')
plt.ylabel('IQ Score')
plt.title('Regression Analysis of IQ Score by Birth Year')
plt.show()

# Print the slope and intercept of the regression line
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
