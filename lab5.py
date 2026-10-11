
import warnings
warnings.filterwarnings("ignore")

# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Step 1: Load Dataset
df = pd.read_csv("Mall_Customers.csv")

print(df.head())
print(df.columns)

# Step 2: Select Features
X = df[['Age',
        'Annual Income (k$)',
        'Spending Score (1-100)']]

# Step 3: Scale the Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Find Optimal K
inertias = []
silhouette_scores = []

K = range(2, 11)

for k in K:
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X_scaled)
    inertias.append(model.inertia_)

    silhouette_scores.append(
        silhouette_score(X_scaled, labels)
    )

# Step 5: Plot Elbow and Silhouette Graphs
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(K, inertias, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(K, silhouette_scores, marker='o')
plt.title("Silhouette Score")
plt.xlabel("Number of Clusters")
plt.ylabel("Score")
plt.grid(True)

plt.tight_layout()
plt.show()

# Step 6: Apply K-Means
best_k = 5

kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(X_scaled)

# Step 7: Cluster Profile
cluster_profile = df.groupby("Cluster")[
    ["Age",
     "Annual Income (k$)",
     "Spending Score (1-100)"]
].mean()

print("\nCluster Profile")
print(cluster_profile)

# Step 8: Scatter Plot
plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    hue="Cluster",
    palette="viridis",
    s=100
)

plt.title("Customer Segmentation using K-Means")
plt.grid(True)
plt.show()

# Step 9: Number of Customers in Each Cluster
plt.figure(figsize=(6, 4))

sns.countplot(
    x="Cluster",
    data=df
)

plt.title("Number of Customers in Each Cluster")
plt.show()

# Step 10: Average Spending Score
plt.figure(figsize=(6, 4))

sns.barplot(
    data=df,
    x="Cluster",
    y="Spending Score (1-100)"
)

plt.title("Average Spending Score by Cluster")
plt.show()