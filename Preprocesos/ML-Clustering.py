from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Cargamos el dataframe de movies limpio, luego del ETL
movies_ml = pd.read_csv('Dataset/movies_ml.csv')
vectores_sinopsis = pd.read_csv('Dataset/vectores_sinopsis.csv')

movies_ml = pd.concat([movies_ml, vectores_sinopsis], axis=1)

# Borramos las columnas innecesarias en el modelo de kmeans
movies_ml = movies_ml.drop(columns = ["id", "title"])


# Método del codo (Elbow Method)
# Consiste en calcular la inercia para distintas instancias del algoritmo de clustering,
# graficarlas y determinar el punto de inflexión más fuerte (codo). 
# Ese será el n de clusters más eficiente

n_clusters = range(10, 70, 5)
elbow_scores = []
silhouette_scores = []

for k in n_clusters:
    kmeans = KMeans(n_clusters = k, n_init = 10)
    kmeans.fit(np.array(movies_ml))
    elbow_scores.append(kmeans.inertia_)

# Graficar la suma de las distancias cuadradas intra-cluster (inertias) en función de k
plt.plot(n_clusters, elbow_scores, 'bx-')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Inercia')
plt.title('Método del Codo')
plt.show()

# Por el gráfico notamos el codo en k = 30 clusters

# Creamos las instancia de KMeans con el valor de cluster óptimo y hacemos las predicciones
kmeans = KMeans(n_clusters = 30, n_init = 10, random_state = 123)
kmeans.fit(movies_ml)
labels = kmeans.predict(movies_ml)

# Exportamos las predicciones para ser consumidas por el modelo de ML
# Convertimos el array de labels a texto y con los valores tipo entero
np.savetxt('Dataset/labels.csv', labels, delimiter = ',', fmt = '%d')  
