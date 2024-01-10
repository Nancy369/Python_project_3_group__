import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler

#get the file of processing data
file_path = './datasets/CVD_cleaned_processing_3.csv'
data = pd.read_csv(file_path)

#choose the correct column and Sex
selected_data = data[['Heart_Disease_New', 'Skin_Cancer_New', 'Other_Cancer_New', 'Depression_New', 'Arthritis_New', 'Diabetes_New', 'Sex']]

male_data = selected_data[selected_data['Sex'] == 'Male']
female_data = selected_data[selected_data['Sex'] == 'Female']

##data standardlization by males
male_scaler = StandardScaler()
scaled_male_data = male_scaler.fit_transform(male_data.iloc[:, :-1])  # 除了性别列之外的特征列进行标准化

##data standardlization by females
female_scaler = StandardScaler()
scaled_female_data = female_scaler.fit_transform(female_data.iloc[:, :-1])  # 除了性别列之外的特征列进行标准化

#use MinBatchKMeans method
num_clusters = 5
male_kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=42)
female_kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=42)
male_cluster_labels = male_kmeans.fit_predict(scaled_male_data)
female_cluster_labels = female_kmeans.fit_predict(scaled_female_data)
data.loc[data['Sex'] == 'Male', 'Cluster_Labels_disease_male'] = male_cluster_labels
data.loc[data['Sex'] == 'Female', 'Cluster_Labels_disease_female'] = female_cluster_labels

output_file_path_male = './cluster_label_male_2.csv'
data[data['Sex'] == 'Male'][['Cluster_Labels_disease_male']].to_csv(output_file_path_male, index=False)

output_file_path_female = './cluster_label_female_2.csv'
data[data['Sex'] == 'Female'][['Cluster_Labels_disease_female']].to_csv(output_file_path_female, index=False)

clustered_data = pd.concat([male_data, female_data])

#descend 3D
pca = PCA(n_components=3)
pca_result = pca.fit_transform(clustered_data.iloc[:, :-2])  # 所有特征列进行PCA降维，最后两列是聚类标签列

fig = plt.figure(figsize=(12, 6))

#Visualizing clustering results for males

ax1 = fig.add_subplot(121, projection='3d')
scatter1 = ax1.scatter(pca_result[clustered_data['Sex'] == 'Male'][:, 0], pca_result[clustered_data['Sex'] == 'Male'][:, 1], pca_result[clustered_data['Sex'] == 'Male'][:, 2], c=male_cluster_labels, cmap='viridis', marker='o', edgecolors='black')
ax1.set_title('Clustering_result_disease- Male',pad = 5)
ax1.legend(*scatter1.legend_elements(), title='Clusters')

#Visualizing clustering results for females
ax2 = fig.add_subplot(122, projection='3d')
scatter2 = ax2.scatter(pca_result[clustered_data['Sex'] == 'Female'][:, 0], pca_result[clustered_data['Sex'] == 'Female'][:, 1], pca_result[clustered_data['Sex'] == 'Female'][:, 2], c=female_cluster_labels, cmap='viridis', marker='o', edgecolors='black')
ax2.set_title('lustering_result_disease- Female',pad = 4)
ax2.legend(*scatter2.legend_elements(), title='Clusters')

plt.tight_layout()
plt.savefig('./results_picture/clustering_result_by_gender_disease.png')
plt.show()

