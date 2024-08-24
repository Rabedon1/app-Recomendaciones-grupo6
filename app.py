from flask import Flask, request, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

app = Flask(__name__)

# Carga de los datos
movies = pd.read_csv('movies.csv')

# Vectorizar los géneros de las películas
tfidf = TfidfVectorizer(stop_words='english')
movies['genres'] = movies['genres'].fillna('')  # Maneja valores nulos
tfidf_matrix = tfidf.fit_transform(movies['genres'])

# Calcular la similitud del coseno entre las películas
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Función para obtener recomendaciones
def get_recommendations(title, cosine_sim=cosine_sim):
    idx = movies[movies['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Obtener las 10 más similares
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices]

@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = []
    if request.method == 'POST':
        pelicula = request.form['pelicula']
        recommendations = get_recommendations(pelicula)
    return render_template('index.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
