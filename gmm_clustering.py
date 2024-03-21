import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn import preprocessing

# Load the data
data = pd.read_csv('Iris.csv')

# Visualize the data
sns.swarmplot(x="Species", y="PetalLengthCm", data=data)
plt.grid()

# Drop unnecessary columns
data = data.drop('Id', axis=1)

# Separate features and target
X = data.iloc[:,0:4]
y = data.iloc[:,-1]

# Standardize features
scaler = preprocessing.StandardScaler()
scaler.fit(X)
X_scaled_array = scaler.transform(X)
X_scaled = pd.DataFrame(X_scaled_array, columns=X.columns)

# Define the number of clusters
nclusters = 3
seed = 0

# Fit Gaussian Mixture Model
gmm = GaussianMixture(n_components=nclusters, random_state=seed)
gmm.fit(X_scaled)

# Predict clusters
y_cluster_gmm = gmm.predict(X_scaled)

# Add cluster labels to dataframe
data['ClusterGMM'] = y_cluster_gmm

# Custom color palette
palette = sns.color_palette("husl", nclusters)  # Choose the palette you prefer

# Visualize clustering results with custom color palette
sns.swarmplot(x="Species", y="PetalLengthCm", hue="ClusterGMM",
palette=palette, data=data)
plt.title('Clustering Results with Gaussian Mixture Model')
plt.grid()
plt.show()