import streamlit as st
import requests
from streamlit_lottie import st_lottie
import joblib
import numpy as np
from PIL import Image


def load_lottie(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


def prepare_input_data_for_model(Gender, age, occupation, mood, movies_you_like, description):
    if Gender == 'Male':
        Gender = 'M'
    else:
        Gender = 'F'

    age= age

    occupation = occupation
    
    mood = mood
    movies_you_like = movies_you_like.split('/')
    description1 = description
    # Convert categorical variables to numerical    

    Features = [Gender, age, occupation, mood, movies_you_like, description1]
    sample = np.array(Features).reshape(-1, len(Features))
    return sample


loaded_model_LR = joblib.load(open("C:/Users/bouma/Python environnement/HCI-project1/course_example/loan_predition_model_LR.pkl", 'rb'))


st.title('Loan Prediction System')
animation_header = load_lottie("https://assets6.lottiefiles.com/packages/lf20_azmc2roh.json")
animation_header2 = load_lottie("https://assets4.lottiefiles.com/packages/lf20_1wnliqn0.json")

st_lottie(animation_header2, speed=1, height=150, key="forth")


lottie_link = "https://assets8.lottiefiles.com/packages/lf20_4wDd2K.json"
animation = load_lottie(lottie_link)
st.write('---')
st.subheader('Please enter your information to predict an ideal movie:')
with st.container():
    right_column, left_column = st.columns(2)
    with right_column:
        Gender = st.radio('Gender:', ['Female', 'Male'])
        age = st.number_input('Age:', value=0)
        occupation = st.selectbox('Occupation:', [
	'academic/educator',
	'artist',
	'clerical/admin', 'college/grad student', 'customer service', 
    'doctor/health care',  
    'executive/managerial', 'farmer', 'homemaker', 'K-12 student',
    'lawyer', 'programmer', 'retired', 'sales/marketing',
    'scientist', 'self-employed', 'technician/engineer',
    'tradesman/craftsman', 'unemployed', 'writer','other or not specified'])
        mood = st.selectbox('Mood:', ['happy', 'sad', 'angry', 'fearful', 'disgusted', 'surprised'])
        movies_you_like = st.text_input('Movies you like (separate each title by "/"):')
        description = st.text_area('Description of a scenario you would like to see:')
       
        

        sample = prepare_input_data_for_model(Gender, age, occupation, mood, movies_you_like, description)
        st.write('---')
        st.subheader('we recommend you theses movies:')
        

    with left_column:
        st_lottie(animation, speed=1, height=400, key="second")


with st.container():
    if st.button('Predict'):
        prediction = loaded_model_LR.predict(sample)
        if prediction == 1:
            st.success("Congratulations! Your loan has been accepted.")
            st.balloons()
        else:
            st.error("Sorry! Your loan has been refused.")

st.write('---')


footer = """<style>
header {visibility: hidden;}

/* Light mode styles */
p {
  color: black;
}

/* Dark mode styles */
@media (prefers-color-scheme: dark) {
  p {
    color: white;
  }
}

a:link , a:visited{
color: #5C5CFF;
background-color: transparent;
text-decoration: none;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

:root {
  --footer-bg-color: #333;
}

@media (prefers-color-scheme: dark) {
  :root {
    --footer-bg-color: rgb(14, 17, 23);
  }
}

@media (prefers-color-scheme: light) {
  :root {
    --footer-bg-color: white;
  }
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: var(--footer-bg-color);
color: black;
text-align: center;
}

</style>
<div class="footer">
<p>&copy; 2023</p>
</div>
"""
st.markdown(footer, unsafe_allow_html=True)
