import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

#get the file
file_path = './datasets/factor_basic_female.csv'
data = pd.read_csv(file_path)

#choose the relevant columns
selected_columns = ['Age_Category_New', 'BMI_New']
X = data[selected_columns]  #independent variable
y = data['General_Health_New']  #dependent variable

imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

imputer_y = SimpleImputer(strategy='mean')
y = imputer_y.fit_transform(y.values.reshape(-1, 1))  #Two-dimensional array
y = y.flatten()  #translate to the one-dimensional array

#use a multiple linear regression model
model = LinearRegression()
model.fit(X, y)
#Retrieve the coefficients of the model
coefficients = model.coef_
#draw the picture
plt.figure(figsize=(8, 6))
plt.bar(selected_columns, coefficients)
plt.title('Coefficients of basical Factors on Female Cardiovascular Diseases Risk Prediction')
plt.xlabel('Factors')
plt.ylabel('Coefficient Value')
plt.savefig('./results_picture/Coefficients of basical Factors on Female Cardiovascular Diseases Risk Prediction')
plt.show()