# Train rain models & test parquet files for music recommendation
# run Music-Recommend-Spotify % python predict_music_popularity.py --genre 'pop'
# run Music-Recommend-Spotify % python predict_music_popularity.py (default genre is rock)


# Import libraries
 
import glob
import pandas as pd
import datetime
import os
import re
import sys
import pandas
from numpy.random import default_rng as rng
from collections import Counter
import numpy as np
from sklearn.neighbors import NearestNeighbors

import argparse

def load_data():
    all_files = glob.glob("music_data/*.csv")
    #df = pd.read_csv("data/filtered_track_df.csv")
    #df = pd.read_csv("data/filtered_track_classic_df.csv")

    li = []

    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        li.append(df)

    df = pd.concat(li, axis=0, ignore_index=True)
    #count total rows
    row_count = len(df)
    #print('ZZZZZZZ ', row_count)
    #remove duplicates based on uri value
    df = df.drop_duplicates(subset=['uri'], keep='first')
    row_count = len(df)
    #print(row_count)
    #df = pd.read_csv(filenames)
    df['genres'] = df.genres.apply(lambda x: [i[1:-1] for i in str(x)[1:-1].split(", ")])
    exploded_track_df = df.explode("genres")
    #print(len(df))
    filtered_df = df
    return exploded_track_df, filtered_df

# Define list of genres and audio features of songs for audience to choose from
genre_names = ['Dance Pop', 'Electronic', 'Electropop', 'Hip Hop', 'Jazz', 'K-pop', 'Latin', 
               'Pop', 'Pop Rap', 'R&B', 'Rock', 'Folk Rock']
audio_feats = ["acousticness", "danceability", "energy", "instrumentalness", "valence", "tempo"]

# Load data
#exploded_track_df, filtered_df = load_data()

# Define function to return Spotify URIs and audio feature values of top neighbors (ascending)
def n_neighbors_uri_audio(exploded_track_df, filtered_df, genre):
    #genre = 'folk rock'
    start_year = 1960
    end_year = 2000
    genre = genre.lower()
    #print(exploded_track_df["genres"])
    #print(exploded_track_df['artists_id'], exploded_track_df['artists_name'])
    genre_data = exploded_track_df[(exploded_track_df["genres"]==genre) & (exploded_track_df["release_year"]>=start_year) & (exploded_track_df["release_year"]<=end_year)]
    
    genre_data = genre_data.sort_values(by='popularity', ascending=False)[:500] # use only top 500 most popular songs
    
    neigh = NearestNeighbors()
    neigh.fit(genre_data[audio_feats].to_numpy())
    
    acousticness = max(filtered_df.acousticness) - min(filtered_df.acousticness)
    danceability = max(filtered_df.danceability) - min(filtered_df.danceability)
    energy = max(filtered_df.energy) - min(filtered_df.energy)
    instrumentalness = max(filtered_df.instrumentalness) - min(filtered_df.instrumentalness)
    valence = max(filtered_df.valence) - min(filtered_df.valence)
    tempo = max(filtered_df.tempo) - min(filtered_df.tempo)
    test_feat = [acousticness, danceability, energy, instrumentalness, valence, tempo]
    #test_feat = [0.5, 0.5, 0.5, 0.0, 0.45, 118.0]
    
    #n_neighbors = neigh.kneighbors([test_feat], n_neighbors=len(genre_data), return_distance=False)[0]
    n_neighbors = neigh.kneighbors([test_feat], n_neighbors=len(genre_data), return_distance=False)[0]
    
    #artists_id = exploded_track_df['artists_id']
    artists_id = genre_data.iloc[n_neighbors]['artists_id'].to_numpy()
    artists_name = genre_data.iloc[n_neighbors]['artists_name'].to_numpy()
    artist_info = [genre_data.iloc[n_neighbors]['artists_id'].to_numpy(), genre_data.iloc[n_neighbors]['artists_name'].to_numpy()]
    uris = genre_data.iloc[n_neighbors]["uri"].tolist()
    audios = genre_data.iloc[n_neighbors][audio_feats].to_numpy()
    #print(artists_name[0])
    print(artist_info)

    return genre_data
#genre_data = n_neighbors_uri_audio(exploded_track_df, filtered_df)

def dataset_parquet_file(genre_data):
    temp_folder_path = '/Users/dineshk/work/MLOps/music/Music-Recommend-Spotify/music_data'
    
    genre_data.rename(columns=lambda x: re.sub(r'[^\w]', '', x), inplace=True)
    # Shuffling the dataset for more accurate training and testing results.
    genre_data = genre_data.sample(frac=1).reset_index(drop=True)
    # Splitting the data up so that 80% of the data is training data, 20% testing data.
    training_data = genre_data[:int(genre_data.shape[0]*.8)]
    testing_data = genre_data[int(genre_data.shape[0]*.8):]
    
    print("Creating diamonds dataset parquet files...")
    with open(os.path.join(temp_folder_path, "train_artists.parquet"), 'w'):
        training_data.to_parquet(os.path.join(temp_folder_path, "train_artists.parquet"))
    with open(os.path.join(temp_folder_path, "test_artists.parquet"), 'w'):
        testing_data.to_parquet(os.path.join(temp_folder_path, "test_artists.parquet"))
    
    print("Artists dataset parquet files created.")

    return genre_data   
#genre_data = dataset_parquet_file(genre_data)

def create_artist_recommend(genre_data):
    temp_folder_path = '/Users/dineshk/work/MLOps/music/Music-Recommend-Spotify/music_data'

    # Saving a CSV file of the dataset for predicting purposes.
    #print(genre_data)
    artist_recommend_file = 'artist_recommend.csv'
    
    csv_predict = genre_data.drop(columns=['popularity'])
    # The number of data points we want to predict on when calling mlflow pyfunc predict.
    num_pred = 50
    csv_predict[:num_pred].to_csv(os.path.join(temp_folder_path, "artists_new.csv"), index=False)
    # This CSV file contains the price of the tested diamonds.
    # Predictions can be compared with these actual values.
    genre_data[["artists_name", "name", "genres", "release_year", "popularity"]][:num_pred].to_csv(os.path.join(temp_folder_path, artist_recommend_file),
                                   index=False)
    
    print("Artists recommend songs file ", artist_recommend_file, " created.")
    
#create_artist_recommend(genre_data)

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--genre", "-g", type=str, default='rock')
     
    parsed_args = args.parse_args()
    
    # Load data
    exploded_track_df, filtered_df = load_data()
    genre_data = n_neighbors_uri_audio(exploded_track_df, filtered_df, parsed_args.genre)
    genre_data = dataset_parquet_file(genre_data)
    create_artist_recommend(genre_data)