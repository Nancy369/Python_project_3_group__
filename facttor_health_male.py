import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

#get the file
file_path = './datasets/factor_health_male.csv'
data = pd.read_csv(file_path)

#choose the relevant columns
selected_columns = ['Checkup_New', 'exercising_New', 'Alcohol_Consumption_New', 'Smoking_History_New', 'Green_Vegetables_Consumption', 'FriedPotato_Consumption_New','Fruit_Consumption']
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
plt.figure(figsize=(10, 10))
plt.bar(selected_columns, coefficients)
plt.title('Coefficients of health Factors on Male Cardiovascular Diseases Risk Prediction')
plt.xlabel('Factors')
plt.xticks(rotation=16)
plt.ylabel('Coefficient Value')
plt.savefig('./results_picture/Coefficients of health Factors on Male Cardiovascular Diseases Risk Prediction')
plt.show()
