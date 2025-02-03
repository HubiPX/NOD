import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. Wczytanie danych Digits
digits = load_digits()
X = digits.data  # wektory 64-pikselowe
y = digits.target  # Prawdziwe etykiety cyfr

# 2. Normalizacja danych
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Klasteryzacja K-Means
kmeans = KMeans(n_clusters=10, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# 4. Redukcja wymiarów do 2D za pomocą PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 5. Wizualizacja klastrów na wykresie 2D
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.6)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            s=200, c='red', marker='X', label='Centroidy')

plt.title("K-Means na zbiorze Digits (redukcja PCA do 2D)")
plt.xlabel("Główna składowa 1")
plt.ylabel("Główna składowa 2")
plt.legend()
plt.show()
