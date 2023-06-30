
# <h1 align=center> **PROYECTO INDIVIDUAL Nº1** </h1>

# <h1 align=center> **Modelo recomendador de películas. FastAPI y deployment en Render** </h1>

## **Introducción**

El proyecto consiste en un modelo recomendador de películas que utiliza técnicas de aprendizaje automático de Clustering para ofrecer sugerencias de películas similares a la película ingresada. El modelo se basa en un algoritmo de agrupamiento (clustering) que utiliza análisis de contenido y procesamiento de lenguaje natural (NLP) para conseguir una buena calidad en las recomendaciones.

## **Objetivo**

Como desarrolladores debemos tener la capacidad de llevar a cabo un proceso completo que abarque desde la concepción de los datos hasta la entrega del producto al cliente, que pueda consumirlo.
Para conseguir la implementación se aplicó un pipeline de datos completo que incluyó la limpieza, procesamiento y transformación de datos, análisis exploratorio de datos (EDA), clustering y el despliegue del servicio de API utilizando FastAPI y Render.

# **Desarrollo**

## ETL

Contamos con dos datasets en formato csv donde está nuestra materia prima para el proyecto. Uno es un dataset de aproximadamente 45 mil películas con 20 features (columnas). El otro dataset es metadata con información relacionada a las películas, aquí encontramos al "cast" (elenco) y "crew" (equipo de filmación) de cada una.

Como primer obstáculo al ingestar el dataset de movies encontramos que tenemos columnas anidadadas en formato json. Son features categóricas multilabel:
- "belongs_to_collection", "genres", "production_companies", "production_countries", "spoken_languages"
Se elaboró una estrategia que desanide y luego haga el encoding de estas features. Se hizo una función que devuelve un dataframe con las categorías de cada fila en una lista y también, a modo informativo para el desarrollador, una lista con los valores únicos de esa variable categórica. A los NaN se los asigno a la misma etiqueta "Sin_categoría", para luego decidir qué hacer con estos datos.
Gracias a esto obtuvimos que el dataset se divide en 20 géneros distintos y 1695 colecciones de películas. 23538 compañías productoras de 160 países distintos se encargaron de la producción, y las películas fueron filmadas en 75 lenguajes distintos.

Este es un resumen de las tareas de ETL. También hubo una importante limpieza de datos y acondicionamiento a los formatos deseados, ejemplo: "Release_date". Se anexo una nueva columna calculada "return", con el retorno de cada película (la relación entre "revenue" y "budget").

## EDA

Luego del ETL se exportó el dataset en csv para ser importado por otra sección. Considero más legible ir haciendo el proyecto en etapas bien divididas.
En el EDA comenzamos tomando decisiones como descartar la feature "belongs_to_collection" ya que sería una gran matriz dispersa con la mayoría de datos en blanco (La colección más grande abarcaba 9 películas). A las variables categóricas multilabel aplicamos un Multilabel Binarizer de Scikit learn para transformarlas a columnas binarias y poder ser consumidas por un algoritmo de ML. Debido a la gran cantidad de compañías de "production_companies", más de 23 mil, optamos por elegir las más relevantes e hicimos una función con el "top 60 de productoras que más filmaron". Lo mismo se hizo con "production_countries" con un top 40 y "spoken_languages" con un top 10. Siempre se buscó que al menos el 50% de las películas estén categorizadas.

Después de esto se seleccionó las variables continuas y se las analizó a cada una por separado. Con un gráfico de violín se pudo ver si tenían outliers o valores atípicos y en caso de ser así, se analizó los casos puntualmente. Al final se concluyó casi todos los valores de estas features eran normales. Luego de tener la certeza de que cada una por separado tenía integridad, se hizo una matriz de correlación para intentar detectar posibles dependencias.

Para agregarle complejidad y un poco más de versatilidad al análisis de contenido, optamos por hacer NLP (procesamiento de lenguaje natural) a la columna "overview", que son las sinopsis de las películas. Con la librería TfidfVectorizer de scikit-learn y un set de stop words (fuente: https://countwordsfree.com/stopwords) transformamos nuestras sinopsis en vectores de números para que puedan ser procesadas por nuestro algoritmo de ML más adelante. TfidfVectorizer hace ponderaciones de las palabras por el número de apariciones, tanto en la propia sinopsis como en el total. Debido a que hay una infinidad de palabras en las sinopsis entrenamos nuestro TfidfVectorizer con las 100 palabras más relevantes. Luego de esto concatenamos los resultados con las demás features.

## ML-Clustering
Ya habiendo analizado los datos pasamos a la sección de machine learning. Tenemos que hacer una buena selección de los hiperparámetros del algoritmo que haga el agrupamiento de nuestras películas.
Elegí el modelo K-Means de Scikit learn utilizando la distancia euclidea para hacer el cómputo, buscar similares y elegir el mejor centroide. Para calibrarlo correctamente usé el método del codo para determinar que con k = 30 clusters se conseguía una buena inercia.

### Main
En el main se ejecuta la API de FastAPI que nos brinda la interfaz para que podamos consumir nuestro recomendador de películas. La aplicación tenía los siguientes endpoints, los primeros cinco en los que tenemos que aplicar transformaciones y cálculo para conseguir el resultado, y el último nuestro recomendador de películas:

+ def **cantidad_filmaciones_mes( *`Mes`* )**:
    Se ingresa un mes en idioma Español. Debe devolver la cantidad de películas que fueron estrenadas en el mes consultado en la totalidad del dataset.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Ejemplo de retorno: *`X` cantidad de películas fueron estrenadas en el mes de `X`*

+ def **cantidad_filmaciones_dia( *`Dia`* )**:
    Se ingresa un día en idioma Español. Debe devolver la cantidad de películas que fueron estrenadas en día consultado en la totalidad del dataset.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Ejemplo de retorno: *`X` cantidad de películas fueron estrenadas en los días `X`*

+ def **score_titulo( *`titulo_de_la_filmación`* )**:
    Se ingresa el título de una filmación esperando como respuesta el título, el año de estreno y el score.
    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Ejemplo de retorno: *La película `X` fue estrenada en el año `X` con un score/popularidad de `X`*

+ def **votos_titulo( *`titulo_de_la_filmación`* )**:
    Se ingresa el título de una filmación esperando como respuesta el título, la cantidad de votos y el valor promedio de las votaciones. La misma variable deberá de contar con al menos 2000 valoraciones, caso contrario, debemos contar con un mensaje avisando que no cumple esta condición y que por ende, no se devuelve ningun valor.
    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Ejemplo de retorno: *La película `X` fue estrenada en el año `X`. La misma cuenta con un total de `X` valoraciones, con un promedio de `X`*

+ def **get_actor( *`nombre_actor`* )**:
    Se ingresa el nombre de un actor que se encuentre dentro de un dataset debiendo devolver el éxito del mismo medido a través del retorno. Además, la cantidad de películas que en las que ha participado y el promedio de retorno. **La definición no deberá considerar directores.**
    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Ejemplo de retorno: *El actor `X` ha participado de `X` cantidad de filmaciones, el mismo ha conseguido un retorno de `X` con un promedio de `X` por filmación*

+ def **get_director( *`nombre_director`* )**:
    Se ingresa el nombre de un director que se encuentre dentro de un dataset debiendo devolver el éxito del mismo medido a través del retorno. Además, deberá devolver el nombre de cada película con la fecha de lanzamiento, retorno individual, costo y ganancia de la misma.