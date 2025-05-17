# Import libraries
import os
import sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def recommend_DL(age, gender, occupation):

    # Reading ratings file
    ratings = pd.read_csv('ratings.csv', sep='\t', encoding='latin-1', 
                        usecols=['user_id', 'movie_id', 'user_emb_id', 'movie_emb_id', 'rating'])
    max_userid = ratings['user_id'].drop_duplicates().max()
    max_movieid = ratings['movie_id'].drop_duplicates().max()

    # Reading ratings file
    users = pd.read_csv('users.csv', sep='\t', encoding='latin-1', 
                        usecols=['user_id', 'gender', 'zipcode', 'age_desc', 'occ_desc'])

    # Reading ratings file
    movies = pd.read_csv('movies.csv', sep='\t', encoding='latin-1', 
                        usecols=['movie_id', 'title', 'genres'])


    RNG_SEED = 42  # You can choose any integer value

    # Create training set
    shuffled_ratings = ratings.sample(frac=1., random_state=RNG_SEED)

    # Shuffling users
    Users = shuffled_ratings['user_emb_id'].values
    print('Users:', Users, ', shape =', Users.shape)

    # Shuffling movies
    Movies = shuffled_ratings['movie_emb_id'].values
    print('Movies:', Movies, ', shape =', Movies.shape)

    # Shuffling ratings
    Ratings = shuffled_ratings['rating'].values
    print('Ratings:', Ratings, ', shape =', Ratings.shape)


    # Import Keras libraries
    from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
    # Import CF Model Architecture
    from CFModel import CFModel

    # Define constants
    K_FACTORS = 100 # The number of dimensional embeddings for movies and users
    TEST_USER = 2000 # A random test user (user_id = 2000)

    # Define model
    model = CFModel(max_userid, max_movieid, K_FACTORS)
    # Compile the model using MSE as the loss function and the AdaMax learning algorithm
    model.compile(loss='mse', optimizer='adamax')


    # Callbacks monitor the validation loss
    # Save the model weights each time the validation loss has improved
    callbacks = [EarlyStopping('val_loss', patience=2), 
                ModelCheckpoint('weights.h5', save_best_only=True)]



    # Use the pre-trained model
    trained_model = CFModel(max_userid, max_movieid, K_FACTORS)
    # Load weights
    trained_model.load_weights('weights.h5')


    # Pick a random test user
    users[users['user_id'] == TEST_USER]


    # Function to predict the ratings given User ID and Movie ID
    def predict_rating(user_id, movie_id):
        return trained_model.rate(user_id - 1, movie_id - 1)



    user_ratings = ratings[ratings['user_id'] == TEST_USER][['user_id', 'movie_id', 'rating']]
    user_ratings['prediction'] = user_ratings.apply(lambda x: predict_rating(TEST_USER, x['movie_id']), axis=1)
    user_ratings.sort_values(by='rating', 
                            ascending=False).merge(movies, 
                                                    on='movie_id', 
                                                    how='inner', 
                                                    suffixes=['_u', '_m']).head(20)
                            
                            
                            
    recommendations = ratings[ratings['movie_id'].isin(user_ratings['movie_id']) == False][['movie_id']].drop_duplicates()
    recommendations['prediction'] = recommendations.apply(lambda x: predict_rating(TEST_USER, x['movie_id']), axis=1)
    recommendations.sort_values(by='prediction',
                            ascending=False).merge(movies,
                                                    on='movie_id',
                                                    how='inner',
                                                    suffixes=['_u', '_m']).head(20)
                            
    return recommendations
