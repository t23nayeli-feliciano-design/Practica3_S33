import pandas as pd
import numpy as np
import joblib
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# 1. Simulación de un dataset multivariado de comportamiento de clientes en comercio electrónico
# 6 Variables: [Gasto Anual, Num_Pedidos, Tiempo_Navegacion, Devoluciones, Soporte_Tickets, Antiguedad]
X_raw, _ = make_blobs(
    n_samples=1200, n_features=6, centers=4, cluster_std=1.2, random_state=42
)

# 2. Estandarización de variables (Z-score Scaling)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# 3. Determinación de K óptimo evaluando métricas
resultados_k = []
for k in range(2, 6):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    lbls = km.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, lbls)
    db = davies_bouldin_score(X_scaled, lbls)
    ch = calinski_harabasz_score(X_scaled, lbls)
    resultados_k.append({
        'K': k,
        'Silhouette': sil,
        'Davies-Bouldin': db,
        'Calinski-Harabasz': ch
    })

df_metricas = pd.DataFrame(resultados_k)

# Selección del mejor K (máximo coeficiente de silueta)
best_k = int(df_metricas.loc[df_metricas['Silhouette'].idxmax()]['K'])

# 4. Entrenamiento del modelo final de K-Means con K óptimo
kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
clusters_finales = kmeans_final.fit_predict(X_scaled)

# 5. Reducción de dimensionalidad con PCA a 2D para compresión y visualización
pca_final = PCA(n_components=2)
X_pca_2d = pca_final.fit_transform(X_scaled)

print("=== PIPELINE COMPLETO DE ANÁLISIS NO SUPERVISADO EN UBUNTU EC2 ===")
print("\n--- Evaluaciones para Selección de K ---")
print(df_metricas.to_string(index=False))
print(f"\n-> Número Óptimo de Clusters Seleccionado: K = {best_k}")

print("\n--- Métricas del Modelo Final Seleccionado ---")
print(f"Coeficiente de Silueta: {silhouette_score(X_scaled, clusters_finales):.4f}")
print(f"Índice Davies-Bouldin: {davies_bouldin_score(X_scaled, clusters_finales):.4f}")
print(f"Índice Calinski-Harabasz: {calinski_harabasz_score(X_scaled, clusters_finales):.4f}")
print(f"Varianza Explicada por PCA 2D: {np.sum(pca_final.explained_variance_ratio_)*100:.2f}%")

# 6. Serialización de modelos en el entorno Ubuntu EC2
joblib.dump(kmeans_final, 'modelo_kmeans_clientes.pkl')
joblib.dump(scaler, 'scaler_clientes.pkl')
joblib.dump(pca_final, 'pca_clientes.pkl')
print("\n-> Modelo K-Means, Escalador y PCA guardados exitosamente como archivos '.pkl'.")
