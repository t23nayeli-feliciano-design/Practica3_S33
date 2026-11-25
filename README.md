# Práctica 3 (S33): Aprendizaje No Supervisado — Segmentación de Clientes y PCA en AWS EC2

## Resultados Obtenidos
* **Número Óptimo de Clusters Seleccionado:** `K = 4`
* **Coeficiente de Silueta (`K = 4`):** `0.7339` (el valor más alto entre las configuraciones evaluadas)
* **Índice Davies-Bouldin (`K = 4`):** `0.3822` (el valor más bajo entre las configuraciones evaluadas)
* **Índice Calinski-Harabasz (`K = 4`):** `3349.18`
* **Varianza Explicada por PCA 2D:** `80.89%`

## Análisis e Interpretación de Resultados
* **Selección del K óptimo (Silueta y Davies-Bouldin):** El valor **`K = 4`** es el óptimo porque maximiza el **Coeficiente de Silueta (0.7339)** —indicando alta cohesión interna dentro de cada grupo y clara separación respecto a los clusters vecinos— y al mismo tiempo minimiza el **Índice Davies-Bouldin (0.3822)**, lo que confirma la menor dispersión intracluster y la mayor distancia entre centroides frente a `K = 2, 3` o `5`.
* **Retención de información con PCA (de 6 a 2 dimensiones):** Al comprimir las 6 variables originales de comportamiento de clientes a únicamente 2 componentes principales (`PCA 2D`), el modelo logró conservar aproximadamente el **80.89% de la varianza total** de los datos, reduciendo dos tercios de la dimensionalidad sin perder la estructura esencial que diferencia a los segmentos.
* **Integración de los tres archivos `.pkl` en producción (API REST / AWS S3):** Al consumir el modelo en tiempo real desde una API o tras descargarlo de un bucket de **AWS S3**, el flujo sigue un orden estricto para cada nuevo cliente:
  1. **`scaler_clientes.pkl`:** Estandariza las 6 métricas crudas del cliente usando exactamente la media y desviación estándar del entrenamiento.
  2. **`modelo_kmeans_clientes.pkl`:** Recibe los datos ya escalados y asigna instantáneamente el segmento o cluster (`0` a `3`) al que pertenece el cliente para personalizar campañas o reglas de negocio.
  3. **`pca_clientes.pkl`:** Proyecta ese mismo vector escalado a coordenadas `(PC1, PC2)` para alimentar tableros de monitoreo en tiempo real o sistemas de visualización 2D sin reprocesar todo el conjunto histórico.
