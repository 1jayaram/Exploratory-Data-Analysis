# Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv(r"C:/Data Analyst/Exploratory Data Analysis/Sample - Superstore.csv", encoding='ISO-8859-1')

# Drop unneeded columns
df.drop(['Postal Code'], axis=1, inplace=True)

# Convert Order Date to datetime
df['Order Date'] = pd.to_datetime(df['Order Date'])
df['Month'] = df['Order Date'].dt.to_period('M')
df['Year'] = df['Order Date'].dt.year

# Summary
print(df.describe())
print(df.isnull().sum())

# Feature Engineering
df['Profit Margin'] = df['Profit'] / df['Sales']
df['Year-Month'] = df['Order Date'].dt.to_period('M').astype(str)

# Monthly Sales Trend
monthly_sales = df.groupby('Year-Month')['Sales'].sum().reset_index()
fig1 = px.line(monthly_sales, x='Year-Month', y='Sales', title='Monthly Sales Trend')
fig1.show()

# Region-wise Profitability
region_profit = df.groupby('Region')['Profit'].sum().reset_index()
fig2 = px.bar(region_profit, x='Region', y='Profit', color='Region', title="Profit by Region")
fig2.show()

# Correlation Heatmap
corr = df[['Sales', 'Quantity', 'Discount', 'Profit']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Hypothesis Testing: Does discount significantly affect profit?
low_discount = df[df['Discount'] < 0.2]['Profit']
high_discount = df[df['Discount'] >= 0.2]['Profit']
t_stat, p_value = stats.ttest_ind(low_discount, high_discount)
print(f"T-stat: {t_stat:.2f}, P-value: {p_value:.4f}")
if p_value < 0.05:
    print("Statistically significant: Discount affects Profit")
else:
    print("No significant difference in profit due to discount levels")

# Predict Future Sales using Linear Regression
# Use monthly aggregation
monthly = df.groupby('Year-Month').agg({'Sales':'sum'}).reset_index()
monthly['Month_Num'] = np.arange(len(monthly))

X = monthly[['Month_Num']]
y = monthly['Sales']

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# Evaluation
print(f"MAE: {mean_absolute_error(y, y_pred):.2f}")
print(f"R²: {r2_score(y, y_pred):.2f}")

# Visualization
plt.figure(figsize=(10, 5))
plt.plot(monthly['Month_Num'], y, label='Actual')
plt.plot(monthly['Month_Num'], y_pred, label='Predicted', linestyle='--')
plt.title("Linear Regression - Sales Forecast")
plt.xlabel("Month Number")
plt.ylabel("Sales")
plt.legend()
plt.tight_layout()
plt.savefig("plot_output.png")  # Choose your desired filename

