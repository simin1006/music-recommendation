import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

st.set_page_config(page_title="Music Recommendation System", page_icon="🎵")

st.title("🎵 Music Recommendation System")

# Load dataset
df = pd.read_csv("cleaned data.csv")

# Features
features = [
    "valence",
    "danceability",
    "energy",
    "tempo",
    "acousticness",
    "instrumentalness",
    "speechiness",
    "popularity",
    "explicit"
]

# Scale data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[features])

# Train model
model = NearestNeighbors(n_neighbors=5, metric="euclidean")
model.fit(scaled_data)

st.header("Enter Music Features")

valence = st.slider("Valence", 0.0, 1.0, 0.5)
danceability = st.slider("Danceability", 0.0, 1.0, 0.5)
energy = st.slider("Energy", 0.0, 1.0, 0.5)
tempo = st.slider("Tempo", 0.0, 250.0, 120.0)
acousticness = st.slider("Acousticness", 0.0, 1.0, 0.5)
instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.0)
speechiness = st.slider("Speechiness", 0.0, 1.0, 0.1)
popularity = st.slider("Popularity", 0, 100, 50)
explicit = st.selectbox("Explicit", [0, 1])

input_data = pd.DataFrame([[

    valence,
    danceability,
    energy,
    tempo,
    acousticness,
    instrumentalness,
    speechiness,
    popularity,
    explicit

]], columns=features)

if st.button("Recommend"):

    input_scaled = scaler.transform(input_data)

    distances, indices = model.kneighbors(input_scaled)

    st.subheader("Recommended Similar Records")

    st.dataframe(df.iloc[indices[0]].reset_index(drop=True))