# Gender-Specific Cardiovascular and Cerebrovascular Impact Factors Explored through Cluster Analysis

Through cluster analysis and Multiple linear regression analysis, we discussed gender-specific cardiovascular and cerebrovascular impact factors.

## The structure of project
- Python_project_3_group__
    - cluster_picture
        - the result picture of lifestyle factor label
        - the result picture of health factor label
        - the result picture of disease factor label
    - datasets
        - CVD_cleared.csv
        - CVD_cleared_processing_3.csv
        - cluster_data_XX.csv
        - factor_XX.csv
    - results picture
        - Coefficients of different Factors on different gender Cardiovascular Diseases Risk.png
        - Coefficients of all clusters on different gender Cardiovascular Diseases Risk.png
    - cluster_data_XX.py
    - factor_XX.py
    - dataprocessing_record.txt
    - valiation.txt

## dataset
We choose **The Behavioral Risk Factor Surveillance System (BRFSS) dataset** as the data sources. This dataset can be found on [https://www.kaggle.com/datasets/alphiree/cardiovascular-diseases-risk-prediction-dataset]. This dataset encompasses numerous indicators related to health-related risk behaviors, chronic health conditions, and utilization of preventive services.

### get the imformation of the dataset
We use **data_processing.py** to get the information of the BRFSS dataset. The result was saved in **Dataprocessing_record**.

## data processing
We assign scores to each option for ease of subsequent calculation, the score is higher,that means the choice is healthier for the people.The score detail is saved in valiation.

## Cluster Analysis
The options are grouped into three major categories based on their relevance， which included **Individual Demographics, Chronic Diseases and Other Health Issues,Lifestyle factors**.The detail is below.

| Group | Include |
| ------ | ------ |
| Individual Demographics | Age Category<br>BMI |
| hronic Diseases and Other Health Issues | Heart Disease<br>Skin Cancer<br>Other Cancer<br>Depression<br>Arthritis<br>diabetes |
| Lifestyle factors | checkup<br>exercise<br>smoking history<br>alcohol consumption<br>Fruit Consumption<br>Green vegetables Consumption<br>Fried potato Consumption |
