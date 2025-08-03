import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dot, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
from numpy import sqrt
import os

# Load MovieLens data
ratings = pd.read_csv('dataa/ml-100k/u.data', sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])
movies = pd.read_csv('dataa/ml-100k/u.item', sep='|', encoding='latin-1', 
                     names=['movie_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL', 
                            'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 
                            'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 
                            'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'])

# Preprocess data
ratings = ratings.drop('timestamp', axis=1)
ratings['user_id'] = ratings['user_id'] - 1  # Zero-indexed
ratings['movie_id'] = ratings['movie_id'] - 1

# Parameters
num_users = ratings['user_id'].nunique()
num_movies = ratings['movie_id'].nunique()
embedding_size = 50

# Build model
def build_model():
    user_input = Input(shape=(1,), name='user_input')
    movie_input = Input(shape=(1,), name='movie_input')
    user_embedding = Embedding(num_users, embedding_size, name='user_embedding')(user_input)
    movie_embedding = Embedding(num_movies, embedding_size, name='movie_embedding')(movie_input)
    user_vec = Flatten()(user_embedding)
    movie_vec = Flatten()(movie_embedding)
    dot_product = Dot(axes=1)([user_vec, movie_vec])
    dot_product = Dropout(0.2)(dot_product)  # Add dropout to prevent overfitting
    output = Dense(1, activation='linear')(dot_product)
    model = Model([user_input, movie_input], output)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Load or train model
model_path = 'movie_recommender_model.h5'
if os.path.exists(model_path):
    print(f"Loading existing model from {model_path}")
    model = tf.keras.models.load_model(model_path)
else:
    print("Training new model...")
    model = build_model()
    early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
    try:
        model.fit([ratings['user_id'].values, ratings['movie_id'].values], ratings['rating'].values,
                  epochs=10, batch_size=64, validation_split=0.2, callbacks=[early_stopping], verbose=1)
        model.save(model_path)
        print(f"Model saved to {model_path}")
    except Exception as e:
        print(f"Error during training: {e}")

# Evaluate model
predictions = model.predict([ratings['user_id'].values, ratings['movie_id'].values])
rmse = sqrt(mean_squared_error(ratings['rating'].values, predictions))
print(f'RMSE: {rmse}')

# Genre-based recommendation function
def recommend_by_genre(user_id, genre, model, movies_df, num_recommendations=5):
    # Validate user_id
    if not (1 <= user_id <= num_users):
        return f"Invalid user_id. Must be between 1 and {num_users}."
    user_id = user_id - 1  # Zero-indexed
    # Validate genre
    valid_genres = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 
                    'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 
                    'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    if genre not in valid_genres:
        return f"Invalid genre. Choose from: {', '.join(valid_genres)}"
    # Filter movies by genre
    genre_movies = movies_df[movies_df[genre].eq(1)]['movie_id'].values - 1
    rated_movies = ratings[ratings['user_id'] == user_id]['movie_id'].values
    unrated_genre_movies = np.setdiff1d(genre_movies, rated_movies)
    if len(unrated_genre_movies) == 0:
        return f"No unrated {genre} movies available for user {user_id + 1}."
    # Predict ratings
    user_array = np.array([user_id] * len(unrated_genre_movies))
    predictions = model.predict([user_array, unrated_genre_movies])
    top_indices = predictions.flatten().argsort()[-num_recommendations:][::-1]
    recommended_movie_ids = unrated_genre_movies[top_indices]
    return movies_df[movies_df['movie_id'].isin(recommended_movie_ids + 1)][['title', genre]]

# Interactive input
def get_user_input():
    try:
        user_id = int(input(f"Enter user ID (1 to {num_users}): "))
        genre = input("Enter genre (e.g., Comedy, Action, Drama): ")
        recommendations = recommend_by_genre(user_id, genre, model, movies)
        print(f"\nRecommendations for User {user_id} in {genre}:")
        print(recommendations)
    except ValueError:
        print("Invalid input. Please enter a valid user ID (numeric).")
    except Exception as e:
        print(f"Error: {e}")

# Run interactive recommendation
if __name__ == "__main__":
    get_user_input()