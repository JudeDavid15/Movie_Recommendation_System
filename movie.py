import streamlit as st
import pickle as pk
from sklearn.metrics.pairwise import cosine_similarity

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins&display=swap');
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }
    .stApp {
        background-color: #960018;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

movie = pk.load(open('movie.pkl', 'rb'))
tfidf_matrix = pk.load(open('tfidf_matrix.pkl', 'rb'))

st.header('Movie Recommender')
selected_movie = st.selectbox('Select movies', movie['names'].values)

def recommend(sel_movie):
    sel_movie = sel_movie.lower()
    idx = movie[movie['names'] == sel_movie].index[0]
    sims = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    dist = sorted(list(enumerate(sims)), reverse=True, key=lambda x: x[1])
    return [movie.iloc[i[0]].names for i in dist[1:6]]

if st.button("Show Recommend"):
    movie_name = recommend(selected_movie)
    cols = st.columns(5)
    for c, name in zip(cols, movie_name):
        with c:
            st.text(name)