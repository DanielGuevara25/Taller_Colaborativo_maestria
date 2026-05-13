# Segmentación de Clientes con Aprendizaje No Supervisado

---

## Descripción del Problema

Este proyecto implementa técnicas de aprendizaje no supervisado para segmentar clientes según características demográficas y patrones de comportamiento de compra.

El objetivo principal es descubrir agrupamientos naturales dentro de los datos utilizando algoritmos de clustering y técnicas de reducción de dimensionalidad.

El dataset contiene información de clientes como:

- Género
- Edad
- Ingreso anual
- Spending Score

Técnicas aplicadas:

- K-means
- DBSCAN
- PCA
- t-SNE

---

# Estructura del Repositorio

```txt
segmentacion-clientes-ml/
│
├── data/
│   └── Mall_Customers.csv          # Dataset original
│
├── notebooks/
│   └── segmentacion_clientes.ipynb # Notebook principal
│
├── outputs/
│   ├── elbow_method.png
│   ├── silhouette_score.png
│   ├── clusters_kmeans.png
│   ├── clusters_dbscan.png
│   ├── pca_visualizacion.png
│   ├── tsne_visualizacion.png
│   └── resumen_clusters.csv
│
├── requirements.txt
├── README.md
└── .gitignore
