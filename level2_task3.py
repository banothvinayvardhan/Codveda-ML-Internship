import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(r"C:\Users\Maharaj\OneDrive\Desktop\Codveda_Internship\churn-bigml-80.csv")

# clustering just based on day vs eve mins
X = df[['Total day minutes', 'Total eve minutes']]
X_scaled = StandardScaler().fit_transform(X)

# figure out optimal k using elbow method
errors = []
for i in range(1, 10):
    km = KMeans(n_clusters=i, random_state=42, n_init=10)
    km.fit(X_scaled)
    errors.append(km.inertia_)

plt.figure(figsize=(12, 5))

# elbow plot
plt.subplot(1, 2, 1)
plt.plot(range(1, 10), errors, marker='o')
plt.title('Elbow Method')
plt.xlabel('Clusters')
plt.ylabel('Error')

# fitting with 4 clusters
final_km = KMeans(n_clusters=4, random_state=42, n_init=10)
clusters = final_km.fit_predict(X_scaled)

print("clustering done.")

# scatter plot
plt.subplot(1, 2, 2)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap='rainbow', alpha=0.5)
plt.scatter(final_km.cluster_centers_[:, 0], final_km.cluster_centers_[:, 1], s=100, c='black', marker='X')
plt.title('K-Means Clusters (k=4)')
plt.xlabel('Day Mins (scaled)')
plt.ylabel('Eve Mins (scaled)')

print("showing graphs...")
plt.show()