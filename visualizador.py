import streamlit as st
from PIL import Image
import pandas as pd
import cv2
import numpy as np
import requests
from io import BytesIO
from PIL import Image as PILImage
import os
import pandas as pd
import requests
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
# ------ Algoritmo de PCA ----------
def update_centroids(X, labels, k):
    # Cálculo eficiente de nuevos centroides para cada feature
    n_features = X.shape[1]
    centroids = np.zeros((k, n_features))
    for i in range(n_features):
        sums = np.bincount(labels, weights=X[:, i], minlength=k)
        counts = np.bincount(labels, minlength=k)
        centroids[:, i] = sums / np.maximum(counts, 1)
    return centroids

def kmeans(X, k=3, max_iters=100, tol=1e-6, random_state=42):
    np.random.seed(random_state)
    n_samples, n_features = X.shape

    # Inicializa centroides seleccionando k muestras aleatorias
    indices = np.random.choice(n_samples, k, replace=False)
    centroids = X[indices]

    for iteration in range(max_iters):
        # Calcula distancias cuadradas usando producto escalar (más eficiente que norm)
        X_norm = np.sum(X**2, axis=1).reshape(-1, 1)          # ||x||^2
        C_norm = np.sum(centroids**2, axis=1).reshape(1, -1)  # ||c||^2
        distances = X_norm + C_norm - 2 * X @ centroids.T     # ||x - c||^2

        # Asigna cada punto al centroide más cercano
        labels = np.argmin(distances, axis=1)

        # Guarda centroides anteriores para verificar convergencia
        old_centroids = centroids.copy()

        # Recalcula centroides: media de los puntos asignados (vectorizado)
        centroids = np.zeros((k, n_features))
        counts = np.bincount(labels, minlength=k)
        for dim in range(n_features):
            sums = np.bincount(labels, weights=X[:, dim], minlength=k)
            centroids[:, dim] = sums / np.maximum(counts, 1)

  
        # Verifica convergencia: si los centroides ya no cambian
        centroid_shifts = np.linalg.norm(centroids - old_centroids, axis=1)
        if np.all(centroid_shifts < tol):
            print(f"Convergió en la iteración {iteration}")
            break

    return labels, centroids

def get_genres(string_):
    if not string_:
        return []
    return [genre.strip() for genre in string_.split("|")]


# ------ Fin de PCA ---------- 
def descargar_imagen(url,  save_path="toystory.png"):
    try:
        # Descargar imagen
        response = requests.get(url)
        response.raise_for_status()  
        with open(save_path, 'wb') as f:
            f.write(response.content)

        print(f"Imagen descargada y guardada como: {save_path}")
    except Exception as e:
        print(f"Error procesando la imagen: {e}")


def get_hsv_histogram_from_image(image, bins=(8, 8, 8)):
    try:
        # img = PILImage.open(BytesIO(image)).convert('RGB')
        img = image.convert('RGB')
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        # Calcular histograma 3D en HSV
        hist = cv2.calcHist([img], [0, 1, 2], None, bins,
                            [0, 180, 0, 256, 0, 256])
        # Normalizar el histograma
        cv2.normalize(hist, hist)
        return hist.flatten()
    except Exception as e:
        print(f"Error procesando la imagen: {e}")
        return np.zeros(np.prod(bins))  # vector vacío si falla

def get_movie_poster(tmdb_id, api_key="281b94a01dd7c4539eb3d0ac5bd067df" ):
    """
    Devuelve la URL del póster de una película usando su tmdb_id y la API de TMDb.
    
    :param tmdb_id: ID de la película en TMDb.
    :param api_key: Tu clave de API de TMDb.
    :return: URL del póster o None si no se encuentra.
    """
    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}"
    params = {
        "api_key": api_key,
        "language": "en-US"
    }
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        poster_path = data.get("poster_path")
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500{poster_path}"
        else:
            print(f"No poster found for TMDb ID {tmdb_id}")
    else:
        print(f"Error {response.status_code} for TMDb ID {tmdb_id}")
    
    return None

def ejecutar_busqueda_por_similitud(X_train_with_data, df_features_hsv, generos_seleccionados, k=10, n_components_=50, top_n=10):
    st.subheader("Suba un póster para buscar películas similares")
    poster = st.file_uploader("Suba un póster", type=["jpg", "png", "jpeg"])
    if poster is not None:
        imagen = Image.open(poster)
        st.image(imagen, caption="Póster subido", width=200)

        poster_feature = get_hsv_histogram_from_image(imagen)

        pca = PCA(n_components=n_components_, random_state=42)
        X_train = df_features_hsv.drop(columns=['movieId', 'title']).values
        X_train_pca = pca.fit_transform(X_train)
        poster_pca = pca.transform(poster_feature.reshape(1, -1))

        labels, centroids = kmeans(X_train_pca, k=k)
        test_cluster = np.argmin(euclidean_distances(poster_pca, centroids), axis=1)[0]
        cluster_indices = np.where(labels == test_cluster)[0]
        cluster_vectors = X_train_pca[cluster_indices]

        similarities = cosine_similarity(poster_pca, cluster_vectors)[0]
        top_10_local = np.argsort(similarities)[-top_n:][::-1]
        top_10_global = cluster_indices[top_10_local]

        st.subheader("Recomendaciones basadas en el póster")
        cols = st.columns(3)
        col_index = 0

        for idx in top_10_global:
            movie = X_train_with_data.iloc[idx]
            # Filtrar por género
            if generos_seleccionados:
                movie_genres = get_genres(movie["genres"])  # Asumo que esta función devuelve lista de géneros
                if not any(g in movie_genres for g in generos_seleccionados):
                    continue  # No mostrar si no tiene ningún género seleccionado

            tmdb_id = X_train_with_data[X_train_with_data["movieId"] == movie["movieId"]]["tmdbId"].values
            if len(tmdb_id) == 0 or np.isnan(tmdb_id[0]):
                continue
            url = get_movie_poster(int(tmdb_id[0]))
            if url:
                with cols[col_index]:
                    st.image(url, caption=movie["title"], use_container_width=True)
                    genres = get_genres(movie["genres"])
                    st.markdown("**Géneros:** " + ", ".join(genres))
                col_index = (col_index + 1) % 3

def ejecutar_busqueda_peliculas_por_cluster(X_train_with_data, df_features_hsv, generos_seleccionados, k=10, top_n=5, n_components_=50):
    st.subheader("Películas representativas por cluster")

    # Procesar y aplicar PCA a las features HSV
    X_train = df_features_hsv.drop(columns=['movieId', 'title']).values
    pca = PCA(n_components=n_components_, random_state=42)
    X_train_pca = pca.fit_transform(X_train)

    # Ejecutar KMeans
    labels, centroids = kmeans(X_train_pca, k=k)

    for cluster_id in range(k):
        st.markdown(f"### Cluster {cluster_id + 1}")
        cluster_indices = np.where(labels == cluster_id)[0]
        cluster_vectors = X_train_pca[cluster_indices]

        # Calcular distancias al centroide
        dists = euclidean_distances(cluster_vectors, centroids[cluster_id].reshape(1, -1)).flatten()
        sorted_indices = cluster_indices[np.argsort(dists)]

        cols = st.columns(5)
        col_index = 0
        count = 0

        for idx in sorted_indices:
            if count >= top_n:
                break

            movie = X_train_with_data.iloc[idx]

            # Filtrar por género
            if generos_seleccionados:
                movie_genres = get_genres(movie["genres"])  # Devuelve lista de géneros
                if not any(g in movie_genres for g in generos_seleccionados):
                    continue  # Si no coincide con ningún género seleccionado, omitir

            tmdb_id = movie.get("tmdbId", None)
            if pd.isna(tmdb_id):
                continue

            url = get_movie_poster(int(tmdb_id))
            if url:
                with cols[col_index]:
                    st.image(url, caption=movie["title"], use_container_width=True)
                    genres = get_genres(movie["genres"])
                    st.markdown("**Géneros:** " + ", ".join(genres))
                col_index = (col_index + 1) % 5
                count += 1



def ejecutar_visualizacion_bidimensional_segun_caracteristicas_visuales(X_train_with_data, df_features_hsv, generos_seleccionados, k=10, top_n=5, n_components_=50):
    st.subheader("Visualización bidimensional de películas según características visuales (HSV)")

    # Extraer features y aplicar PCA
    X_train = df_features_hsv.drop(columns=['movieId', 'title']).values
    pca = PCA(n_components=n_components_, random_state=42)
    X_train_pca = pca.fit_transform(X_train)

    # Reducir a 2D para visualizar
    pca_2d = PCA(n_components=2, random_state=42)
    X_2d = pca_2d.fit_transform(X_train_pca)

    # Ejecutar KMeans
    labels, centroids = kmeans(X_train_pca, k=k)

    # Mapeo para el DataFrame
    df_plot = pd.DataFrame({
        "x": X_2d[:, 0],
        "y": X_2d[:, 1],
        "title": df_features_hsv["title"],
        "cluster": labels
    })

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.scatterplot(data=df_plot, x="x", y="y", hue="cluster", palette="tab10", legend="full", s=50, ax=ax)
    ax.set_title("Distribución de películas en 2D por HSV (PCA) y clusters")
    ax.set_xlabel("Componente principal 1")
    ax.set_ylabel("Componente principal 2")
    st.pyplot(fig)

    # Opción: seleccionar un cluster para ver películas
    selected_cluster = st.selectbox("Seleccione un cluster para ver sus películas representativas:", range(k))
    cluster_indices = np.where(labels == selected_cluster)[0]
    cluster_vectors = X_train_pca[cluster_indices]

    # Calcular distancias al centroide
    dists = euclidean_distances(cluster_vectors, centroids[selected_cluster].reshape(1, -1)).flatten()
    sorted_indices = cluster_indices[np.argsort(dists)]

    st.markdown(f"### Películas representativas del cluster {selected_cluster}")
    cols = st.columns(5)
    col_index = 0
    count = 0

    for idx in sorted_indices:
        if count >= top_n:
            break

        movie = X_train_with_data.iloc[idx]

        # Filtrar por género
        if generos_seleccionados:
            movie_genres = get_genres(movie["genres"])  # Devuelve lista de géneros
            if not any(g in movie_genres for g in generos_seleccionados):
                continue  # No mostrar si no coincide con géneros seleccionados

        tmdb_id = movie.get("tmdbId", None)
        if pd.isna(tmdb_id):
            continue

        url = get_movie_poster(int(tmdb_id))
        if url:
            with cols[col_index]:
                st.image(url, caption=movie["title"], use_container_width=True)
                genres = get_genres(movie["genres"])
                st.markdown("**Géneros:** " + ", ".join(genres))
            col_index = (col_index + 1) % 5
            count += 1



def ejecutar_visualizacion_tridimensional_segun_caracteristicas_visuales(X_train_with_data, df_features_hsv, generos_seleccionados, k=10, top_n=5, n_components_=50):
    st.subheader("Visualización tridimensional de películas según características visuales (HSV)")

    # Extraer features y aplicar PCA
    X_train = df_features_hsv.drop(columns=['movieId', 'title']).values
    pca = PCA(n_components=n_components_, random_state=42)
    X_train_pca = pca.fit_transform(X_train)

    # Reducir a 3D para visualizar
    pca_3d = PCA(n_components=3, random_state=42)
    X_3d = pca_3d.fit_transform(X_train_pca)

    # Ejecutar KMeans
    labels, centroids = kmeans(X_train_pca, k=k)

    # Preparar DataFrame para plotly
    df_plot = pd.DataFrame({
        "x": X_3d[:, 0],
        "y": X_3d[:, 1],
        "z": X_3d[:, 2],
        "title": df_features_hsv["title"],
        "cluster": labels.astype(str)  # para colores categóricos
    })

    fig = px.scatter_3d(df_plot, x='x', y='y', z='z',
                        color='cluster',
                        hover_name='title',
                        title="Distribución 3D de películas por HSV (PCA) y clusters",
                        color_discrete_sequence=px.colors.qualitative.Safe)

    fig.update_layout(margin=dict(l=0, r=0, b=0, t=40))

    st.plotly_chart(fig, use_container_width=True)

    # Selección del cluster para mostrar películas representativas
    selected_cluster = st.selectbox("Seleccione un cluster para ver sus películas representativas:", range(k))
    cluster_indices = np.where(labels == selected_cluster)[0]
    cluster_vectors = X_train_pca[cluster_indices]

    # Calcular distancias al centroide y ordenar
    dists = euclidean_distances(cluster_vectors, centroids[selected_cluster].reshape(1, -1)).flatten()
    sorted_indices = cluster_indices[np.argsort(dists)]

    st.markdown(f"### Películas representativas del cluster {selected_cluster}")
    cols = st.columns(5)
    col_index = 0
    count = 0

    for idx in sorted_indices:
        if count >= top_n:
            break

        movie = X_train_with_data.iloc[idx]

        # Filtrar por géneros seleccionados
        if generos_seleccionados:
            movie_genres = get_genres(movie["genres"])  # Se asume que devuelve lista de géneros
            if not any(g in movie_genres for g in generos_seleccionados):
                continue

        tmdb_id = movie.get("tmdbId", None)
        if pd.isna(tmdb_id):
            continue

        url = get_movie_poster(int(tmdb_id))
        if url:
            with cols[col_index]:
                st.image(url, caption=movie["title"], use_container_width=True)
                genres = get_genres(movie["genres"])
                st.markdown("**Géneros:** " + ", ".join(genres))
            col_index = (col_index + 1) % 5
            count += 1


def main():
    generos_disponibles = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 
                          'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'IMAX', 
                          'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

    # Menú lateral
    st.sidebar.title("Menu")
    seleccion = st.sidebar.selectbox(
        "Selecciona una opción",
        ["Búsqueda por similitud", "Películas por clúster", "Visualización bidimensional", "Visualización tridimensional"]
    )
    st.sidebar.markdown("### Parámetros")
    k = st.sidebar.number_input("k (grupos o clusteres)", min_value=2, max_value=100, value=10, step=1)
    top_n = st.sidebar.number_input("top-n (top recomendaciones)", min_value=1, max_value=50, value=5, step=1)
    r_dimen = st.sidebar.number_input("n-componentes (cantidad de componentes)", min_value=1, max_value=100, value=50, step=1)


    # Carga de datos
    status_placeholder = st.empty()
    # status_placeholder.write("Recopilando data ...")

    try:
        X_train_with_data = pd.read_pickle("train_with_links.pkl")
        df_features_hsv = pd.read_pickle("hsv_features_bins_8_.pkl")
        # status_placeholder.success("Data recopilada correctamente")
    except Exception as e:
        status_placeholder.error(f"Error recopilando data: {e}")
        return

    # Elección de sección
    generos_seleccionados = st.multiselect(
    "Filtrar por género(s)", 
    generos_disponibles, 
    default=generos_disponibles  
)
    if seleccion == "Búsqueda por similitud":
        ejecutar_busqueda_por_similitud(X_train_with_data, df_features_hsv, generos_seleccionados, k=k, top_n =top_n, n_components_= r_dimen )

    elif seleccion == "Películas por clúster":
        ejecutar_busqueda_peliculas_por_cluster(X_train_with_data, df_features_hsv, generos_seleccionados, k=k, top_n=top_n, n_components_= r_dimen)

    elif seleccion == "Visualización bidimensional":
        ejecutar_visualizacion_bidimensional_segun_caracteristicas_visuales(X_train_with_data, df_features_hsv,generos_seleccionados,  k=k, top_n=top_n, n_components_= r_dimen)

    elif seleccion == "Visualización tridimensional":
        ejecutar_visualizacion_tridimensional_segun_caracteristicas_visuales(X_train_with_data, df_features_hsv, generos_seleccionados, k=k, top_n=top_n, n_components_= r_dimen)


main()
