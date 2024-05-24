# Building a regression model that predicts medical insurance cost

# Import the dependencies or libraries

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import warnings
warnings.filterwarnings('ignore')

# Import the data from a csv file into a pandas dataframe

data = pd.read_csv('https://raw.githubusercontent.com/ripatanny101/Medical-Insurance-Cost-Prediction/main/medical_insurance.csv')

# Taking a peek at the data

data.head()

# Data cleaning

# Checking for missing values
data.isnull().sum()

# There are no missing values in the dataset. So, no need for performing data.dropna()

# Convert Categorical Variables to Numeric.
# Need to encode categorical variables such as 'sex', 'smoker', and 'region' to include them in a correlation analysis

data['sex'] = data['sex'].map({'female': 0, 'male': 1})
data['smoker'] = data['smoker'].map({'no': 0, 'yes': 1})
data['region'] = pd.Categorical(data['region']).codes  # Encode region as ordered numeric codes

# Generating a Correlation Matrix Including Categorical Encodings"""

# Compute the correlation matrix for all variables
corr_matrix = data.corr()

# Generate a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix of All Variables')
plt.show()

"""Visualize Data Distributions and Relationships"""

# Histogram of Ages

sns.histplot(data['age'], kde=True)
plt.title('Distribution of Ages')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# A histogram to see if younger or older individuals tend to smoke more, potentially affecting insurance costs.


g = sns.FacetGrid(data, hue='smoker', height=4, aspect=2)
g.map(sns.histplot, 'age', kde=False, binwidth=1)  # Adding binwidth for better visualization
g.add_legend(title='Smoker')
plt.title('Age Distribution by Smoking Status')  # Setting title correctly
plt.xlabel('Age')  # Setting X-axis label
plt.ylabel('Count')  # Setting Y-axis label
plt.show()

# Histogram of Charges (Target Variable)

sns.histplot(data['charges'], kde=True)
plt.title('Distribution of Charges')
plt.xlabel('Charges')
plt.ylabel('Frequency')
plt.show()

# histogram to visualize the distribution of BMI values among the insured population

sns.histplot(data['bmi'], kde=True, color='green')
plt.title('Distribution of BMI')
plt.xlabel('BMI')
plt.ylabel('Frequency')
plt.show()

# Boxplot of Charges by Smoker Status

sns.boxplot(x='smoker', y='charges', data=data)
plt.title('Charges by Smoker Status')
plt.xlabel('Smoker (0=No, 1=Yes)')
plt.ylabel('Charges')
plt.show()

# Scatter Plot of BMI vs. Charges

sns.scatterplot(x='bmi', y='charges', hue='smoker', data=data)
plt.title('BMI vs. Charges Colored by Smoker Status')
plt.xlabel('BMI')
plt.ylabel('Charges')
plt.show()

# Pair Plot of Numerical Features

plt.figure(figsize=(15, 10))
sns.pairplot(data, hue='smoker')
plt.title('Pair Plot of Numerical Features')
plt.show()

# A boxplot to explore how the number of children affects the insurance charges.

sns.boxplot(x=data['children'], y=data['charges'])
plt.title('Insurance Charges by Number of Children')
plt.xlabel('Number of Children')
plt.ylabel('Charges')
plt.show()

# A bar plot to compare the average insurance costs across different regions

mean_charges_by_region = data.groupby('region')['charges'].mean().sort_values()
mean_charges_by_region.plot(kind='bar', color='lightgreen')
plt.title('Average Insurance Charges by Region')
plt.xlabel('Region')
plt.ylabel('Average Charges')
plt.xticks(rotation=45)
plt.show()

# A scatter plot to visualize the interaction between age and charges, colored by smoking status.

sns.scatterplot(x='age', y='charges', hue='smoker', data=data, palette=['red', 'green'], alpha=0.6)
plt.title('Impact of Age and Smoking Status on Charges')
plt.xlabel('Age')
plt.ylabel('Charges')
plt.legend(title='Smoker')
plt.show()

# Divide Data into Training and Test Sets

from sklearn.model_selection import train_test_split

# Separate Features and Target
X = data.drop('charges', axis=1)
y = data['charges']

# Split the Data into Training and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# Perform Standardization

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the Model Training data using Linear regression and evaluating

from sklearn.linear_model import LinearRegression

LR = LinearRegression()
LR.fit(X_train_scaled, y_train)

# Make Predictions Using Your Test Data

y_pred = LR.predict(X_test_scaled)

# Calculate the coefficient

coefficients = LR.coef_
print(f'Coefficients {coefficients}')

# calculate the Intercept

intercept = LR.intercept_
print(f'Intercept {intercept}')

# Assess the Model

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

print("R-squared:", r2_score(y_test, y_pred))
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error:", mean_squared_error(y_test, y_pred, squared=False))

# Deploy the Model

import pickle  #Save the trained model using pickle.

pickle.dump(LR, open('model.pkl', 'wb'))
