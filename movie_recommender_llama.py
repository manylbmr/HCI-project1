


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
    df = pd.read_csv('netflix_titles.csv')
    import ollama
    response = ollama.chat(model='llama3', messages=[{'role': 'user', 'content': 'Hello!'}])
    print(response['message']['content'])


    import faiss
    import numpy as np
    import requests

    df["textual_representation"]=df.apply(create_textual_representation, axis=1)


    dim =4096
    index = faiss.IndexFlatL2(dim)




    index = faiss.read_index('index')

    res = requests.post("http://localhost:11434/api/embeddings",
                        json={'model':'llama3',
                            'prompt': movie_desc,})
    embedding = res.json()['embedding']
    embedding = np.array(embedding, dtype='float32')
    D, I = index.search(embedding.reshape(1, -1), 5)
    print("Top 5 similar movies:")
    best_matches= np.array(df['textual_representation'])[I.flatten()]
    for i, match in enumerate(best_matches):
        print(f"Match {i+1}: {match}")
        print("--------------------------------------------------")
    return best_matches[:3]


