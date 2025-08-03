# 🤖 Movie Recommendation System 
A movie recommendation system using neural collaborative filtering (NCF) to predict user ratings and recommend films based on genre preferences, built with the MovieLens 100K dataset. 
🎓 **Project by**: Yasmine Saad  

🎯 **Objective**  
Design and implement a system capable of:  
- Predicting movie ratings for users using a neural collaborative filtering model.  
- Recommending unrated movies in a user-specified genre (e.g., Comedy, Action, Drama).  
- Providing an interactive command-line interface to input `user_id` and genre.  
- Evaluating model accuracy using Root Mean Squared Error (RMSE).  

🧠 **System Architecture**  
- **Data Loading**: Uses MovieLens 100K dataset (`u.data` for ratings, `u.item` for movie metadata).  
- **Preprocessing**: Adjusts `user_id` (1–943) and `movie_id` (1–1682) to zero-based indexing, removes unused columns.  
- **NCF Model**: Neural network with embedding layers (50 dimensions) for users and movies, followed by a dot product and dense layer to predict ratings.  
- **Genre-Based Recommendation**: Filters movies by genre, predicts ratings for unrated movies, and recommends the top 5.  
- **Evaluation**: Computes RMSE to measure prediction accuracy.  

🛠️ **Technologies Used**  
- 📊 **Dataset**: MovieLens 100K (`u.data`, `u.item`)  
- 🧠 **Python 3.11** with:  
  - `tensorflow` (NCF model)  
  - `pandas` (data manipulation)  
  - `numpy` (numerical operations)  
  - `scikit-learn` (RMSE calculation)  
- 🧰 **Windows 10/11** + VS Code  
- ⚙️ **Virtual Environment**: `ml_env` for dependency management  

🚦 **How It Works**  
1. **Data Loading**:  
   - `u.data` provides `user_id` (1–943, unique user identifier), `movie_id` (1–1682, unique movie identifier), and `rating` (1–5).  
   - `u.item` provides movie titles and genres (e.g., Comedy, Sci-Fi).  
2. **Model Training**:  
   - Loads `movie_recommender_model.h5` if available; otherwise, trains an NCF model with early stopping to prevent overfitting.  
   - Uses `user_id` and `movie_id` embeddings to predict ratings.  
3. **Recommendation**:  
   - User inputs a `user_id` (e.g., 1) and a genre (e.g., Comedy).  
   - System filters movies by genre, excludes rated movies, predicts ratings, and recommends the top 5 unrated movies.  
4. **Evaluation**: Calculates RMSE (e.g., ~0.93) to assess prediction accuracy.  

**Example Output**:  
```plaintext
Training new model...
Epoch 1/10
1250/1250 ━━━━━━━━━━━━━━━━━━━━ 3s 2ms/step - loss: 6.2590 - val_loss: 1.2259
...
Model saved to movie_recommender_model.h5
RMSE: 0.93
Enter user ID (1 to 943): 1
Enter genre (e.g., Comedy, Action, Drama): Comedy
Recommendations for User 1 in Comedy:
                     title  Comedy
100  Back to the Future (1985)     1
127     Groundhog Day (1993)     1
...
```

⌨️ **Controls**  
- Enter a `user_id` (1–943) and a genre (e.g., Comedy, Action, Sci-Fi).  
- Press Enter to view recommendations.  
- Handles errors (e.g., invalid `user_id` or genre).  


📁 **Project Structure**  
```
├── recommender.py                         # Main recommendation script
├── requirements.txt                      # Dependency list
├── .gitignore                            # Files to ignore in Git
├── README.md                             # Project documentation
├── data/ml-100k/                         # MovieLens 100K dataset
│   ├── u.data                            # User ratings
│   ├── u.item                            # Movie metadata
├── movie_recommender_model.h5            # Trained model (generated after execution)
```


📬 **Contact**  
📧 yassminesaad75@gmail.com 
🔗 [LinkedIn – Yasmine Saad](https://www.linkedin.com/in/yasmine-saad-397749278/)  

