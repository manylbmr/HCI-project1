import streamlit as st
import requests
from streamlit_lottie import st_lottie
import joblib
import numpy as np
import re
from PIL import Image
import pandas as pd 
from movie_recommender_llama import recommend 
from movie_recommender_DL import recommend_DL


# Load lottie animations
def load_lottie(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()





    
    



def parse_movie_string(movie_data):
    """
    Parse a movie string or dictionary into a structured format.
    If the input is a string, it splits it into key-value pairs based on the presence of ':'.
    """
    if isinstance(movie_data, str):
        # Processing for string values
        data = {}
        parts = [p.strip() for p in movie_data.split('\n') if p.strip()]
        for part in parts:
            if ':' in part:
                key, val = part.split(':', 1)
                data[key.strip()] = val.strip()
        return data
    elif isinstance(movie_data, dict):
        return movie_data
    else:
        # For other types (like Ellipsis)
        return {
            'Title': 'Unknown Movie',
            'Description': 'No description available',
            'Type': 'Movie',
            'Rating': 'N/A'
        }


def get_movie_recommendations(query):
    try:
        raw_results = recommend(query)
        
        # Remove None, empty strings, and Ellipsis
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

# Form section
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

# Reccommendation button
if st.button('Get Recommendations'):
    # Prepare request
    query = f"{movies_you_like} {description} {mood} {Gender} {age} {occupation}"
    
    # Model call
    recommendations = get_movie_recommendations(query)
    
    
    
    
    
    
    if not recommendations:
        st.warning("No valid recommendations could be generated.")
    else:
        for i, movie in enumerate(recommendations, 1):
            similarity_score = float(movie.get('similarity', 0))
            with st.expander(f"{i}. {movie.get('title', 'Unknown Movie')} (Similarity: {similarity_score:.2%})"):
                st.write(f"**Year:** {movie.get('release_year', 'N/A')}")
                st.write(f"**Rating:** {movie.get('rating', 'N/A')}")
                st.write(f"**Genre:** {movie.get('listed_in', 'N/A')}")
                st.write(f"**Director:** {movie.get('director', 'N/A')}")
                st.write(f"**Cast:** {movie.get('cast', 'N/A')}")
                st.write(f"**Description:** {movie.get('description', 'No description available')}")
    
#     # Display of results
#     st.subheader("We recommend these movies:")
    
#     for i, movie in enumerate(recommendations, 1):
#         print(movie)
#         with st.expander(f"{i}. {movie['title']} (Similarity: {movie['similarity']:.0%})"):
#             st.write(f"**Genre:** Action/Sci-Fi") 
#             st.write(f"**Rating:** 8.8/10")
#             st.write("**Description:** A mind-bending thriller about...")

# # Footer
# footer = """<style>..."""
# st.markdown(footer, unsafe_allow_html=True)


