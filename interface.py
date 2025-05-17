import streamlit as st
import requests
from streamlit_lottie import st_lottie
import joblib
import numpy as np
import re
from PIL import Image
import pandas as pd  # Pour gérer les résultats
from movie_recommender_llama import recommend 
from movie_recommender_DL import recommend_DL# Assurez-vous que cette fonction est correctement importée


# Chargement des animations Lottie
def load_lottie(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()





    
    



def parse_movie_string(movie_data):
    """Parse les données de film qu'elles soient sous forme de chaîne ou de dictionnaire"""
    if isinstance(movie_data, str):
        # Traitement pour les chaînes de caractères
        data = {}
        parts = [p.strip() for p in movie_data.split('\n') if p.strip()]
        for part in parts:
            if ':' in part:
                key, val = part.split(':', 1)
                data[key.strip()] = val.strip()
        return data
    elif isinstance(movie_data, dict):
        # Si c'est déjà un dictionnaire
        return movie_data
    else:
        # Pour les autres types (comme Ellipsis)
        return {
            'Title': 'Unknown Movie',
            'Description': 'No description available',
            'Type': 'Movie',
            'Rating': 'N/A'
        }


def get_movie_recommendations(query):
    try:
        raw_results = recommend(query)  # Votre fonction originale
        
        # Filtrage des résultats invalides
        valid_results = [r for r in raw_results if r and not isinstance(r, type(...))]
        
        return [parse_movie_string(m) for m in valid_results]
    except Exception as e:
        st.error(f"Error getting recommendations: {str(e)}")
        return []

# Interface Streamlit
st.title('Movie Recommendation System')
animation_header = load_lottie("https://assets6.lottiefiles.com/packages/lf20_azmc2roh.json")
animation_header2 = load_lottie("https://assets4.lottiefiles.com/packages/lf20_1wnliqn0.json")

st_lottie(animation_header2, speed=1, height=150, key="forth")

# Section de formulaire
st.write('---')
st.subheader('Please enter your information to get movie recommendations:')

with st.container():
    right_column, left_column = st.columns(2)
    
    with right_column:
        Gender = st.radio('Gender:', ['Female', 'Male'])
        age = st.number_input('Age:', min_value=12, max_value=100, value=25)
        occupation = st.selectbox('Occupation:', [
            'academic/educator', 'artist', 'clerical/admin', 
            'college/grad student', 'customer service', 
            'doctor/health care', 'executive/managerial', 'farmer','homemaker' ,'K-12 student' ,'lawyer' ,
            'programmer' ,'retired' , 'sales/marketing' ,'scientist' ,
            'self-employed' ,'technician/engineer' ,'tradesman/craftsman','unemployed','writer'
        ])
        mood = st.selectbox('Mood:', ['happy', 'sad', 'angry', 'excited', 'romantic'])
        movies_you_like = st.text_input('Movies you like (separate with commas):')
        description = st.text_area('Describe what kind of movie you want to watch:')
        
        st.write('---')
        
    with left_column:
        st_lottie(animation_header, speed=1, height=400, key="second")

# Bouton de recommandation
if st.button('Get Recommendations'):
    # Préparation de la requête
    query = f"{movies_you_like} {description} {mood} {Gender} {age} {occupation}"
    
    # Appel au modèle
    recommendations = get_movie_recommendations(description)
    
    
    
    
    
    
    if not recommendations:
        st.warning("No valid recommendations could be generated.")
    else:
        for movie in recommendations:
            with st.expander(f"{movie.get('Title', 'Unknown Movie')}"):
                st.write(f"**Year:** {movie.get('Release Year', 'N/A')}")
                st.write(f"**Rating:** {movie.get('Rating', 'N/A')}")
                st.write(f"**Genre:** {movie.get('Type', 'N/A')}")
                st.write(f"**Director:** {movie.get('Director', 'N/A')}")
                st.write(f"**Cast:** {movie.get('Cast', 'N/A')}")
                st.write(f"**Description:** {movie.get('Description', 'No description available')}")
    
    # Affichage des résultats
    st.subheader("We recommend these movies:")
    
    for i, movie in enumerate(recommendations, 1):
        with st.expander(f"{i}. {movie['Title']} (Similarity: {movie['Similarity']:.0%})"):
            st.write(f"**Genre:** Action/Sci-Fi")  # À adapter avec vos données réelles
            st.write(f"**Rating:** 8.8/10")
            st.write("**Description:** A mind-bending thriller about...")
        
    
    
    
    

# Footer (conservé de votre version originale)
footer = """<style>..."""
st.markdown(footer, unsafe_allow_html=True)


