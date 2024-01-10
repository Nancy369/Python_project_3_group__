import pandas as pd
import numpy as np

# read the file
file_path = '.\datasets\CVD_cleaned.csv'
output_file_preprocessing = '.\Dataprocessing_record.txt'
data = pd.read_csv(file_path,header = 0)

# find the num of options
with open(output_file_preprocessing,'w') as file:
    for column in data.columns:
        num_different_options = data[column].value_counts()
        line = f"{column},num_different_options：{num_different_options}\n"
        '''file.write(f"{column}\n")'''
        file.write(str(num_different_options))
        file.write("\n\n")

#Assigning values for convenient standardization
health_mapping = {
    'Very Good': 5,
    'Good': 4,
    'Excellent': 3,
    'Fair': 2,
    'Poor': 1
}

checkup_mapping = {
    'Within the past year': 5,
    'Within the past 2 years': 4,
    'Within the past 5 years': 3,
    '5 or more years ago': 2,
    'Never': 1
}

exercising_mapping = {
    'No': 0,
    'Yes': 1
}

Heart_Disease_mapping = {
    'No': 1,
    'Yes': 0
}

Skin_Cancer_mapping = {
    'No': 1,
    'Yes': 0
}

Other_Cancer_mapping = {
    'No': 1,
    'Yes': 0
}

Depression_mapping = {
    'No': 1,
    'Yes': 0
}

Diabetes_mapping = {
    'No': 1,
    'Yes': 0,
    'No, pre-diabetes or borderline diabetes': 0.5,
    'Yes, but female told only during pregnancy': 0
}

Arthritis_mapping = {
    'No': 1,
    'Yes': 0,

}

Sex_mapping = {
    'Female': 1,
    'Male': 0
}

Age_Category_mapping = {
    '65-69': 0,
    '60-64': 0,
    '70-74': 0,
    '55-59': 1,
    '50-54': 1,
    '80+'  : 0,
    '40-44': 1,
    '45-49': 1,
    '75-79': 0,
    '35-39': 2,
    '18-24': 2,
    '30-34': 2,
    '25-29': 2
}

conditions_BMI = [
    (data['BMI'] > 24),
    (data['BMI'] < 18.5),
    ((data['BMI'] >= 18.5) & (data['BMI'] <= 24))
]
choices = [0, 1, 2]

Smoking_History_mapping = {
    'No': 1,
    'Yes': 0
}

Alcohol_Consumption_Max = data['Alcohol_Consumption'].max()

FriedPotato_Consumption_Max = data['FriedPotato_Consumption'].max()

data['General_Health_New'] = data['General_Health'].map(health_mapping)

data['Checkup_New'] = data['Checkup'].map(checkup_mapping)

data['exercising_New'] = data['Exercise'].map(exercising_mapping)

data['Heart_Disease_New'] = data['Heart_Disease'].map(Heart_Disease_mapping)

data['Skin_Cancer_New'] = data['Skin_Cancer'].map(Skin_Cancer_mapping)

data['Other_Cancer_New'] = data['Other_Cancer'].map(Other_Cancer_mapping)

data['Depression_New'] = data['Depression'].map(Depression_mapping)

data['Diabetes_New'] = data['Diabetes'].map(Diabetes_mapping)

data['Arthritis_New'] = data['Arthritis'].map(Arthritis_mapping)

data['Sex_New'] = data['Sex'].map(Sex_mapping)

data['Age_Category_New'] = data['Age_Category'].map(Age_Category_mapping)

data['BMI_New'] = None

data['BMI_New'] = np.select(conditions_BMI, choices, default=None)

data['Smoking_History_New'] = data['Smoking_History'].map(Smoking_History_mapping)

data['Alcohol_Consumption_New'] = Alcohol_Consumption_Max - data['Alcohol_Consumption']

data['FriedPotato_Consumption_New'] = FriedPotato_Consumption_Max - data['FriedPotato_Consumption']

data.to_csv('./datasets/CVD_cleaned_processing_3.csv', mode='a', index=False, header=True)