


def create_textual_representation(row):
    textual_representation = f"""Type: {row['type']}
        Title: {row['title']}
        Director: {row['director']}
        Cast: {row['cast']}
        Country: {row['country']}
        Release Year: {row['release_year']}
        Rating: {row['rating']}
        Duration: {row['duration']}
        Description: {row['description']} """
    return textual_representation


def recommend (movie_desc):
    
    import pandas as pd
    import faiss
    import numpy as np
    import requests
    
    import streamlit as st 
    
    df = pd.read_csv('netflix_titles.csv')
    import ollama
    response = ollama.chat(model='llama3', messages=[{'role': 'user', 'content': 'Hello!'}])
    print(response['message']['content'])


    

    df["textual_representation"]=df.apply(create_textual_representation, axis=1)


    dim =4096
    index = faiss.IndexFlatL2(dim)


    index = faiss.read_index('index')

    res = requests.post("http://localhost:11434/api/embeddings",
                        json={'model':'llama3',
                            'prompt': movie_desc,})
    embedding = res.json()['embedding']
    embedding = np.array(embedding, dtype='float32')
    
    distances, index = index.search(embedding.reshape(1, -1), 5)

    results = []
    for i, (dist, idx) in enumerate(zip(distances[0], index[0])):
        # st.write(f"Match {dist+1}: {df.iloc[idx].title} - Similarity: {dist:.4f}")
        similarity = 1 / (1+dist)
        
        movie_info = df.iloc[idx].to_dict()
        movie_info['similarity'] = similarity
        # print(f"Match {i+1}: {match} - Similarity: {dist:.4f}")
        # print("--------------------------------------------------")
        results.append(movie_info)
      
    return results


