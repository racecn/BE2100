import statsmodels.api as sm
import pandas as pd
from statsmodels.formula.api import ols
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns


data = {
    'birth_year': [1971, 1971, 1971, 1974, 1976, 1970, 1972, 1975, 1975, 1974, 1973, 1972, 1975, 1977, 1977, 1979, 1978, 1978, 1979, 1981, 1984, 1984, 1987, 1988, 1982, 1983, 1988, 1989, 1983, 1982, 1985, 1986, 1980, 1981, 1981, 1986, 1984, 1983, 1987],
    'iq_score': [102, 107, 109, 112, 113, 110, 89, 87, 115, 80, 94, 97, 101, 112, 98, 92, 87, 114, 118, 118, 116, 93, 89, 107, 112, 100, 97, 84, 92, 111, 86, 106, 115, 91, 109, 97, 104, 89, 108]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Fit a linear regression model
X = df[['birth_year']]
y = df['iq_score']
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

# Fit the OLS model
model_ols = ols('iq_score ~ birth_year', data=df).fit()

# Create ANOVA table
anova_table = sm.stats.anova_lm(model_ols, typ=2)
print(anova_table)


# Perform F-test for overall model significance
print("\nF-test (overall model significance):")
print("F-value:", anova_table['F'][0])
print("p-value:", anova_table['PR(>F)'][0])

# Perform t-tests for coefficients
print("\nT-tests (coefficients):")
print(model_ols.summary())

sns.boxplot(x=df['iq_score'])
plt.xlabel('IQ Score')
plt.title('Box Plot of IQ Score')
plt.show()