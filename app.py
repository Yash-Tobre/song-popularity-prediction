import streamlit as st
import joblib
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import os
from dotenv import load_dotenv
from sklearn.decomposition import PCA

st.markdown("""
    <style>
    /* Background color and text color */

    .main {
        background-color: #191414; /* Spotify black */
        color: white; /* White text for contrast */
    }
    /* Heading and subheading styling */
    h1, h2, h3, h4, h5, h6 {
        color: #1DB954; /* Spotify green */
    }
    /* Text input and button styling */
    input, textarea, select {
        color: black;
        background-color: #1DB954; /* Spotify green */
        border: none;
        padding: 8px;
        border-radius: 5px;
        margin: 8px 0px;
        font-size: 16px;
    }
    button {
        color: white;
        background-color: #1DB954; /* Spotify green */
        padding: 10px 20px;
        font-size: 16px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
    }
    /* Dataframe styling */
    .dataframe {
        background-color: white;
        color: black;
        border-radius: 5px;
        padding: 10px;
    }
    /* Prediction output styling */
    .prediction-text {
        color: #1DB954; /* Spotify green for emphasis */
        font-size: 24px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)


load_dotenv()
SPOTIFY_CLIENT_ID = st.secrets("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = st.secrets("SPOTIFY_CLIENT_SECRET")

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=SPOTIFY_CLIENT_ID, client_secret=SPOTIFY_CLIENT_SECRET))


# Loading the model
model = joblib.load('song_popularity_model.pkl')
pca = joblib.load('pca_model.pkl')
kmeans_model = joblib.load('kmeans_model.pkl')
scaler = joblib.load('scaler.pkl')

# Set up Spotify API access
# Function to get song features from Spotify
def get_song_features(track_name, artist_id):
    # Search for the track by name and artist ID
    results = sp.search(q=f"track:{track_name} artist:{artist_id}", type='track', limit=1)
    if not results['tracks']['items']:
        return None

    # Get the track ID
    track_id = results['tracks']['items'][0]['id']
    
    # Get audio features
    features = sp.audio_features(track_id)[0]
    if not features:
        return None
    
    # Extract required fields for prediction
    feature_data = {
        'Duration (ms)': features['duration_ms'],
        'Danceability': features['danceability'],
        'Speechiness': features['speechiness'],
        'Acousticness': features['acousticness'],
        'Instrumentalness': features['instrumentalness'],
        'Liveness': features['liveness'],
        'Valence': features['valence'],
        'Tempo': features['tempo'],
        'Energy': features['energy'],
        'Loudness': features['loudness']
    }
    # Apply PCA for 'Energy' and 'Loudness' to create the 'energy_loudness_pca' feature
    energy_loudness_pca = pca.fit_transform([[feature_data['Energy'], feature_data['Loudness']]])
    feature_data['energy_loudness_pca'] = energy_loudness_pca[0][0]

    del feature_data['Energy']
    del feature_data['Loudness']


    # Add Release Date Ordinal
    release_date = results['tracks']['items'][0]['album']['release_date']
    release_date_ordinal = pd.to_datetime(release_date).toordinal()
    feature_data['Release Date ordinal'] = release_date_ordinal

    return pd.DataFrame([feature_data])

# Prediction threshold function
def classify_popularity(score):
    if score == 0:
        return "Not Popular. This means that if you are a diehard fan you might remember the song but it probably would not make it to the charts."
    elif score==1:
        return "Moderately Popular. This song maybe will make it to the lower half of the charts for a while before fading away, but you might have loved it!"
    else:
        return "Highly Popular. This song will be on the charts and probably in tons of custom playlist! You might see it blow up!"

# Streamlit app layout
st.title("Song Popularity Prediction App")

# Input for Track Name and Artist ID
track_name = st.text_input("Enter Track Name:")
artist_id = st.text_input("Enter Artist ID:")

# Predict button
if st.button("Predict Popularity"):
    if not track_name or not artist_id:
        st.write("Please enter both the track name and artist ID.")
    else:
        # Get song features from Spotify
        song_features = get_song_features(track_name, artist_id)
        
        if song_features is None:
            st.write("Could not find song features. Please check the track name and artist ID.")
        else:
            # Display song features for the user to understand more about the song
            st.subheader("Song Feature Data")
            st.write("Here are the audio features used for prediction:")
            st.dataframe(song_features)

            scaled_features = scaler.transform(song_features[['Duration (ms)', 'Danceability', 'Speechiness',
                                                              'Acousticness', 'Instrumentalness', 'Liveness',
                                                              'Valence', 'Tempo', 'energy_loudness_pca', 
                                                              'Release Date ordinal']])
            
            cluster = kmeans_model.predict(scaled_features)
            popularity_score = cluster[0]
            st.write(f'Popularity Score is {popularity_score}')
            popularity_class = classify_popularity(popularity_score)
            # Calculate the mean popularity score for each cluster
  


            
            # Display results
            st.subheader("Popularity Prediction")
            st.write(f"Popularity Class: {popularity_class}")
