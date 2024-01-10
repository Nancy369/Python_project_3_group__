import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

#get the file of processing data
file_path = './datasets./CVD_cleaned_processing_3.csv'
data = pd.read_csv(file_path)

#choose the correct column and Sex
selected_data = data[['Age_Category_New', 'BMI', 'Sex']]

#data standardlization
scaler = StandardScaler()
scaled_data = scaler.fit_transform(selected_data[['Age_Category_New', 'BMI']])  # 仅标准化数值列

#divide by Sex
male_data = scaled_data[selected_data['Sex'] == 'Male']
female_data = scaled_data[selected_data['Sex'] == 'Female']

#use MinBatchKMeans method_Male
k = 5
minibatch_kmeans_male = MiniBatchKMeans(n_clusters=k, batch_size=6000, n_init=5)
minibatch_kmeans_male.fit(male_data)
data_male = selected_data[selected_data['Sex'] == 'Male']
data.loc[selected_data['Sex'] == 'Male', 'Cluster_Labels_basic_male'] = minibatch_kmeans_male.labels_

#use MinBatchKMeans method_Female
minibatch_kmeans_female = MiniBatchKMeans(n_clusters=k, batch_size=6000, n_init=5)
minibatch_kmeans_female.fit(female_data)
data_female = selected_data[selected_data['Sex'] == 'Female']
data.loc[selected_data['Sex'] == 'Female', 'Cluster_Labels_basic_female'] = minibatch_kmeans_female.labels_

data_with_clusters = pd.concat([data_male, data_female])

#save the label
output_csv_path_male = './cluster_label_male_1.csv'
data.loc[selected_data['Sex'] == 'Male', 'Cluster_Labels_basic_male'] = minibatch_kmeans_male.labels_
data.loc[selected_data['Sex'] == 'Male', ['Cluster_Labels_basic_male']].to_csv(output_csv_path_male)

output_csv_path_female = './cluster_label_female_1.csv'
data.loc[selected_data['Sex'] == 'Female', 'Cluster_Labels_basic_female'] = minibatch_kmeans_female.labels_
data.loc[selected_data['Sex'] == 'Female', ['Cluster_Labels_basic_female']].to_csv(output_csv_path_female)

plt.figure(figsize=(10, 5))
#Visualizing clustering results for males
plt.subplot(1, 2, 1)
plt.scatter(data_male['Age_Category_New'], data_male['BMI'], c=minibatch_kmeans_male.labels_, cmap='viridis')
plt.scatter(minibatch_kmeans_male.cluster_centers_[:, 0], minibatch_kmeans_male.cluster_centers_[:, 1], marker='o', s=20, c='red', label='Centroids')
plt.title('Clustering_result_basic- Male')
plt.legend()

# Visualizing clustering results for females
plt.subplot(1, 2, 2)
plt.scatter(data_female['Age_Category_New'], data_female['BMI'], c=minibatch_kmeans_female.labels_, cmap='viridis')
plt.scatter(minibatch_kmeans_female.cluster_centers_[:, 0], minibatch_kmeans_female.cluster_centers_[:, 1], marker='o', s=20, c='red', label='Centroids')
plt.title('MClustering_result_basic- Female')
plt.legend()

plt.tight_layout()
plt.savefig('./results_picture/clustering_result_by_gender.png')
plt.show()
