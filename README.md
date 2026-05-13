# Segmentación de Clientes con Aprendizaje No Supervisado

---

# Descripción del Problema

Este proyecto implementa técnicas de aprendizaje no supervisado para segmentar clientes utilizando variables demográficas y patrones de comportamiento de compra.

El objetivo principal es identificar agrupamientos naturales dentro del dataset mediante algoritmos de clustering y técnicas de reducción de dimensionalidad.

El dataset contiene información de 200 clientes con las siguientes variables:

- CustomerID
- Gender
- Age
- Annual Income (k$)
- Spending Score (1-100)

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
│   └── Mall_Customers.csv
│
├── notebooks/
│   └── segmentacion_clientes.ipynb
│
├── outputs/
│   ├── metodo_codo.png
│   ├── silhouette_score.png
│   ├── clusters_pca.png
│   ├── clusters_tsne.png
│   └── resumen_clusters.csv
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

# Metodología

## 1. Análisis Exploratorio de Datos (EDA)

Se realizó:

- Inspección general del dataset
- Estadísticas descriptivas
- Verificación de valores nulos
- Análisis de variables categóricas y numéricas

Código utilizado:

```python
df.info()
df.describe()
df.isnull().sum()
```

Resultado obtenido:

- 200 registros
- 5 columnas
- No existen valores nulos
- Variables numéricas y categóricas correctamente identificadas

---

## 2. Preprocesamiento de Datos

Se realizaron las siguientes transformaciones:

- Eliminación de CustomerID
- Conversión de Gender a valores numéricos
- Escalamiento con StandardScaler

Código:

```python
df = df.drop("CustomerID", axis=1)

encoder = LabelEncoder()
df["Gender"] = encoder.fit_transform(df["Gender"])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)
```

---

# Implementación de Modelos

---

## K-means Clustering

Para determinar el número óptimo de clusters se utilizaron:

- Método del codo
- Silhouette Score

Código:

```python
inertias = []
silhouettes = []

K = range(2, 11)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    inertias.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))
```

---

## Método del Codo

Resultado:

- La curva comienza a estabilizarse alrededor de 5 clusters.
- Esto indica que 5 grupos representan adecuadamente la estructura de los datos.

Interpretación:

El método del codo permitió identificar una disminución significativa de la inercia hasta aproximadamente 5 clusters, después de lo cual la mejora comienza a reducirse.

---

## Silhouette Score

Resultado:

- El score aumenta progresivamente conforme aumentan los clusters.
- El mejor valor se aproxima entre 8 y 10 clusters.

Interpretación:

Aunque el silhouette score mejora con más clusters, se seleccionó 5 clusters para mantener un equilibrio entre interpretabilidad y segmentación.

---

## Aplicación Final de K-means

Código:

```python
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)

df["Cluster_KMeans"] = kmeans.fit_predict(X_scaled)
```

---

# DBSCAN

DBSCAN fue utilizado para detectar agrupamientos basados en densidad y posibles anomalías.

Código:

```python
dbscan = DBSCAN(eps=0.9, min_samples=5)

df["Cluster_DBSCAN"] = dbscan.fit_predict(X_scaled)
```

Resultados obtenidos:

| Cluster | Cantidad |
|---|---|
| 1 | 110 |
| 0 | 77 |
| -1 (Ruido) | 13 |

Interpretación:

DBSCAN identificó dos agrupamientos principales y detectó 13 registros considerados ruido o anomalías.

---

# PCA (Principal Component Analysis)

Se utilizó PCA para reducir la dimensionalidad y visualizar los clusters en 2 dimensiones.

Código:

```python
pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)
```

Visualización:

```python
plt.scatter(X_pca[:,0], X_pca[:,1], c=df["Cluster_KMeans"])
```

Interpretación:

PCA permitió observar una separación moderada entre clusters, facilitando la representación visual de los grupos generados por K-means.

---

# t-SNE

Se utilizó t-SNE para visualizar agrupamientos complejos de manera no lineal.

Código:

```python
tsne = TSNE(n_components=2, perplexity=30, random_state=42)

X_tsne = tsne.fit_transform(X_scaled)
```

Interpretación:

t-SNE mostró una separación mucho más clara entre los grupos, evidenciando patrones complejos que PCA no logra representar completamente.

---

# Resumen Estadístico por Cluster

Código:

```python
cluster_summary = df.groupby("Cluster_KMeans").mean()
```

Resultados:

| Cluster | Edad Promedio | Ingreso Promedio | Spending Score |
|---|---|---|---|
| 0 | 32.69 | 86.53 | 82.12 |
| 1 | 36.48 | 89.51 | 18.00 |
| 2 | 49.81 | 49.23 | 40.06 |
| 3 | 24.90 | 39.72 | 61.20 |
| 4 | 55.71 | 53.68 | 36.77 |

---

# Interpretación de Clusters

| Cluster | Perfil Detectado |
|---|---|
| 0 | Clientes premium con alto gasto |
| 1 | Clientes con altos ingresos y bajo gasto |
| 2 | Clientes de comportamiento promedio |
| 3 | Clientes jóvenes con gasto moderado-alto |
| 4 | Clientes mayores con gasto moderado |

---

# Comparación de Modelos

| Modelo | Ventajas | Limitaciones |
|---|---|---|
| K-means | Fácil interpretación y segmentación clara | Requiere definir número de clusters |
| DBSCAN | Detecta ruido y anomalías | Sensible a eps y min_samples |
| PCA | Reducción rápida y visualización simple | Solo detecta relaciones lineales |
| t-SNE | Excelente representación visual | Mayor costo computacional |

---

# Librerías Utilizadas

```txt
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
```

---

# Instalación

## Clonar repositorio

```bash
git clone URL_DEL_REPOSITORIO
```

## Crear entorno virtual

```bash
python -m venv .venv
```

## Activar entorno virtual

### Windows

```bash
.venv\Scripts\activate
```

### Linux

```bash
source .venv/bin/activate
```

## Instalar dependencias

```bash
pip install -r requirements.txt
```

## Ejecutar notebook

```bash
jupyter notebook
```

---

# Conclusiones

Los algoritmos de aprendizaje no supervisado permitieron identificar patrones y perfiles de clientes sin utilizar etiquetas previas.

K-means generó una segmentación clara y fácilmente interpretable, mientras que DBSCAN permitió detectar registros atípicos y agrupamientos por densidad.

PCA facilitó la visualización lineal de los clusters, mientras que t-SNE proporcionó una representación visual más precisa de la separación entre grupos.

Los resultados obtenidos demuestran cómo las técnicas de clustering pueden utilizarse para comprender mejor el comportamiento de los clientes y apoyar estrategias de negocio basadas en datos.

---

# Recomendaciones

- Probar diferentes valores de clusters.
- Ajustar parámetros de DBSCAN.
- Incorporar nuevas variables de comportamiento.
- Implementar técnicas adicionales de clustering.
- Aplicar modelos predictivos complementarios.

---

# Autor

Proyecto académico desarrollado para el análisis de aprendizaje no supervisado y segmentación de clientes.
