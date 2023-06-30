import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import ast
from unidecode import unidecode
from fastapi import FastAPI

# Cargamos el dataframe de movies no escalado, luego del ETL, para las primeras funciones
movies_api = pd.read_csv('movies_aux.csv', parse_dates = ['release_date'],
                         usecols = ["id", "title", "release_year", "release_date", "popularity", 
                                    "vote_average", "vote_count", "return", "budget", "revenue"])

# Cargamos credits limpio, que esta expandido en dos df distintos (uno para el elenco (cast) y otro para el equipo de filmación (crew))
credits_cast = pd.read_csv('credits_cast.csv')
credits_crew = pd.read_csv('credits_crew.csv')

# Creamos una instancia de FastAPI
app = FastAPI()

# Función "cantidad_filmaciones_mes"

def mes_to_num(mes):
    meses = ['enero', 'febrero', 'marzo', 'abril', 'mayo', 'junio', 
            'julio', 'agosto', 'septiembre', 'octubre', 'noviembre', 'diciembre']
    indice_mes = None
    if mes in meses:
        for index, valor in enumerate(meses):
            if mes == valor:
                indice_mes = index+1
                return indice_mes
    else:
        raise ValueError("El mes ingresado no es válido.")
    
@app.get('/cantidad_filmaciones_mes/{mes}')
def cantidad_filmaciones_mes(mes):
    if isinstance(mes, str):
        mes = mes.strip().lower()
        mes_int = mes_to_num(mes)
        filmaciones_mes = int(movies_api["release_date"].apply(lambda x: True if x.month == mes_int else False).sum())
        #return f"{filmaciones_mes} fueron estrenadas en el mes de {mes}"
        return {'mes': mes, 'cantidad': filmaciones_mes}
    else:
        raise ValueError("El tipo de datos ingresado no es válido")
    
# Función cantidad_filmaciones_dia

def dia_to_num(dia):
    dias = ['lunes', 'martes', 'miercoles', 'jueves', 'viernes', 'sabado', 'domingo']
    indice_dia = None
    if dia in dias:
        for index, valor in enumerate(dias):
            if dia == valor:
                indice_dia = index
                return indice_dia
    else:
        raise ValueError("El día ingresado no es válido.")

@app.get('/cantidad_filmaciones_dia/{dia}')
def cantidad_filmaciones_dia(dia):
    if isinstance(dia, str):
        dia = unidecode(dia.strip().lower())
        dia_int = dia_to_num(dia)
        filmaciones_dia = int(movies_api["release_date"].apply(lambda x: True if x.weekday() == dia_int else False).sum())
        # return f"{filmaciones_dia} fueron estrenadas en los días {dia}"
        return {'dia': dia, 'cantidad': filmaciones_dia}
    else:
        raise ValueError("El tipo de datos ingresado no es válido")

# Función score_titulo

@app.get('/score_titulo/{titulo_film}')
def score_titulo(titulo_film: str):
    movie = movies_api.loc[movies_api["title"].isin([titulo_film])]
    if not movie.empty:
        resultados = []
        for index, row in movie.iterrows():
            resultados.append({'titulo': titulo_film, 'anio': row['release_year'], 'popularidad': row['popularity']})
        return resultados
    else:
        return "No se encontró la película"

# Función votos_titulo

@app.get('/votos_titulo/{titulo_film}')
def votos_titulo(titulo_film: str):
    movie = movies_api.loc[movies_api["title"].isin([titulo_film])]
    if not movie.empty:
        resultados = []
        for index, row in movie.iterrows():
            if row["vote_count"] > 2000:
                resultados.append({'titulo': titulo_film, 'anio': row['release_year'], 'voto_total': int(row['vote_count']), 'voto_promedio': row['vote_average']})
            else:
                resultados.append({'titulo': titulo_film, 'anio': row['release_year'], 'mensaje': "No cumple con al menos 2000 valoraciones"})
        return resultados
    else:
        return "No se encontró la película"

def metadata(peliculas):        # Función auxiliar para get_actor y get_director
    df_peliculas = movies_api.loc[movies_api["id"].isin(peliculas)]
    retornos = df_peliculas["return"].to_list()
    nombres = df_peliculas["title"].to_list()
    años_estreno = df_peliculas["release_year"].to_list()
    budgets = df_peliculas["budget"].to_list()
    revenues = df_peliculas["revenue"].to_list()
    return retornos, nombres, años_estreno, budgets, revenues

# Función get_actor

@app.get('/get_actor/{nombre_actor}')
def get_actor(nombre_actor: str):
    try:
        peliculas_actor_id = []
        last_id = None
        peliculas_actor_id = ast.literal_eval(credits_cast.loc[credits_cast["cast_name"] == nombre_actor, "id_movie"].values[0])
        retornos, *_ = metadata(peliculas_actor_id)
        retorno_total = sum(retornos)
        cantidad_peliculas_actor = len(peliculas_actor_id)
        retorno_promedio = round(retorno_total/cantidad_peliculas_actor, 2)
        retorno_total = round(retorno_total, 2)
        return {'actor': nombre_actor, 'cantidad_filmaciones': cantidad_peliculas_actor,
                'retorno_total': retorno_total, 'retorno_promedio': retorno_promedio}
    except Exception as e:
        return f"Se produjo un error, tal vez no se encontró el actor. Error: {str(e)}"
    
# Función get_director

@app.get('/get_director/{nombre_director}')
def get_director(nombre_director: str):
    try:
        peliculas_director_id = []
        last_id = None
        peliculas_director_id = ast.literal_eval(credits_crew.loc[credits_crew["crew_name"] == nombre_director, "id_movie"].values[0])
        retornos, peliculas, años_estreno, budgets, revenues = metadata(peliculas_director_id)
        retorno_total = round(sum(retornos), 2)
        retornos = [round(num, 2) for num in retornos]
        budgets = [round(num, 2) for num in budgets]
        revenues = [round(num, 2) for num in revenues]
        return {
                'director': nombre_director,
                'retorno_total': retorno_total,
                'peliculas': [
                            {
                            "titulo": pelicula,
                            'anio': anio,
                            'retorno': retorno,
                            'budget': budget,
                            'revenue': revenue
                            }
                            for pelicula, anio, retorno, budget, revenue in zip(peliculas, años_estreno, retornos, budgets, revenues)
                            ]
                }
    except Exception as e:
        return f"Se produjo un error, tal vez no se encontró el director. Error: {str(e)}"

# Machine Learning. Clustering

# Cargamos el dataframe de movies transformado y escalado, luego del ETL
movies_ml = pd.read_csv('movies_ml.csv')
vectores_sinopsis = pd.read_csv('vectores_sinopsis.csv')
 
# Borramos las columnas innecesarias en el modelo de kmeans
movies_ml = movies_ml.drop(columns = ["id", "title"])
movies_ml = pd.concat([movies_ml, vectores_sinopsis], axis = 1)

# Importamos las predicciones de KMeans que se hicieron anteriormente
labels = np.loadtxt('labels.csv', delimiter=',', dtype = int)

@app.get('/recomendacion/{titulo_film}')
def recomendacion(titulo_film: str):
    '''Ingresas un nombre de pelicula y te recomienda las similares en una lista'''
    rows = movies_ml.loc[movies_api["title"] == titulo_film]
    titulos_similares_general = []
    for indice, row in rows.iterrows():
        cluster_referencia = labels[indice]
        # Encontrar todas las películas en el mismo cluster que el punto de referencia
        peliculas_cluster = movies_ml.iloc[labels == cluster_referencia]
        # Calcular las distancias entre el punto de referencia y todas las películas del cluster
        distancias = np.linalg.norm(peliculas_cluster - row.values, axis = 1)
        # Ordenar las películas por distancia y seleccionar las 5 más cercanas
        indices_peliculas_similares = np.argsort(distancias)[1:6]
        peliculas_similares = peliculas_cluster.iloc[indices_peliculas_similares]
        # Obtener los títulos de las películas similares
        titulos_similares = [movies_api.loc[movies_api.index == index_pelicula]['title'].values[0]
                             for index_pelicula in peliculas_similares.index]
        titulos_similares_general.append({'lista recomendada': titulos_similares})
    return titulos_similares_general


