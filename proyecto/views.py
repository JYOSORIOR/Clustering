from django.shortcuts import render
import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import io
import base64
import urllib

def plot_to_base64(plt):
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    uri = 'data:image/png;base64,' + urllib.parse.quote(string)
    buf.close()
    return uri

def limpiar_y_preprocesar(df, features):
    df = df[features]
    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors='coerce')
    df.fillna(df.mean(), inplace=True)
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    return df

def obtener_grafico_dispersion(df):
    plt.figure()
    sns.scatterplot(x='Mortalidad_infantil', y='PIB_per_cápita', data=df, palette='rainbow')
    return plot_to_base64(plt)

def obtener_grafico_codo(X):
    scc = []
    for i in range(1, 11):
        modelo = KMeans(n_clusters=i, random_state=0, n_init=10)
        modelo.fit(X)
        scc.append(modelo.inertia_)
    plt.figure()
    plt.plot(range(1, 11), scc)
    plt.xlabel('Clusters')
    plt.ylabel('Inercia')
    plt.title('Método del Codo')
    return plot_to_base64(plt)

def clustering(X):
    kmeans = KMeans(n_clusters=3, random_state=0, max_iter=300, n_init=10)
    clusters = kmeans.fit_predict(X)
    return clusters

def obtener_grafico_cluster(df, clusters):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Mortalidad_infantil', y='PIB_per_cápita', hue='Cluster', data=df, palette='viridis', legend='full', edgecolor='k', s=100)
    plt.title('Clustering de Países')
    plt.xlabel('Mortalidad Infantil')
    plt.ylabel('PIB per cápita')
    plt.grid(True)
    return plot_to_base64(plt)

def tablas_cluster(df, clusters):
    cluster_tables = {}
    for cluster_id in range(len(set(clusters))):
        cluster_df = df[df['Cluster'] == cluster_id]
        cluster_tables[cluster_id] = cluster_df
    return cluster_tables

def calcular_silhouette_score(X, clusters):
    score = silhouette_score(X, clusters)
    return score

def upload_csv(request):
    error_carga = False
    exito_carga = False
    datos_csv = []
    df = None
    original_df = None  
    grafico_dispersion = None
    grafico_codo = None
    grafico_cluster = None
    tablas_clusters = {}
    score_silhouette = None

    if request.method == 'POST' and request.FILES.get('csv_file'):
        archivo_csv = request.FILES['csv_file']
        if archivo_csv.name.endswith('.csv'):
            datos = archivo_csv.read().decode('utf-8')
            datos_csv = list(csv.reader(datos.splitlines()))
            exito_carga = True

            df = pd.DataFrame(datos_csv[1:], columns=datos_csv[0])
            original_df = df.copy()  
    
            caracteristicas = ['Mortalidad_infantil', 'Ingreso_neto_per_cápita', 'Esperanza_de_vida_al_nacer', 'Gasto_en_salud', 'PIB_per_cápita']
            
            # Verificar si todas las columnas requeridas están en el df
            if not set(caracteristicas).issubset(df.columns):
                error_carga = True
                exito_carga = False
            else:
                # Limpiar y preprocesar los datos
                df = limpiar_y_preprocesar(df, caracteristicas)

                # Proceso de clustering
                X = df[caracteristicas]

                # Gráfico de dispersión
                grafico_dispersion = obtener_grafico_dispersion(df)

                # Gráfico del codo
                grafico_codo = obtener_grafico_codo(X)

                clusters = clustering(X)
                df['Cluster'] = clusters
                
                original_df['Cluster'] = clusters

                # Gráfico del cluster
                grafico_cluster = obtener_grafico_cluster(df, clusters)

                # Gráfico de las tablas con el cluster
                tablas_clusters = tablas_cluster(original_df, clusters) 

                score_silhouette = calcular_silhouette_score(X, clusters)
        else:
            error_carga = True
        
    contexto = {
        'error_carga': error_carga,
        'exito_carga': exito_carga,
        'datos_csv': datos_csv,
        'grafico_dispersion': grafico_dispersion,
        'grafico_codo': grafico_codo,
        'grafico_cluster': grafico_cluster,
        'tablas_clusters': tablas_clusters,
        'score_silhouette': score_silhouette
    }

    return render(request, 'upload_csv.html', contexto)
