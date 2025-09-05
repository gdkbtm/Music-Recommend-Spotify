# Import libraries
import glob
import pandas as pd
import datetime
from sklearn.neighbors import NearestNeighbors
import streamlit.components.v1 as components
from numpy.random import default_rng as rng
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('TkAgg') # or 'Qt5Agg'
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist

import os
import re
import sys
import pandas

# Define list of genres and audio features of songs for audience to choose from
audio_feats = ["acousticness", "danceability", "energy", "instrumentalness", "valence", "tempo"]

def read_data():
    #global df  # Declare df as global within the function
    all_files = glob.glob("csv_data/*.csv")
    list_contact_csv = []
    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        #add all csv data
        list_contact_csv.append(df) 
    df = pd.concat(list_contact_csv, axis=0, ignore_index=True)

    return df

def load_data(name):
    filtered_df = []
    df = read_data()
    #if df is not None:
    df = df.drop_duplicates(subset=['uri'], keep='first')
    #add new column
    if(len(name) > 0):
        df['artists_name_lower'] = df['artists_name'].str.lower()
        #df['artists_name'] = df['artists_name'].str.lower()
        row_count = len(df)
        print(row_count)
    
        filtered_df = df[df['artists_name_lower'] == name.lower()]
        df['genres'] = df.genres.apply(lambda x: [i[1:-1] for i in str(x)[1:-1].split(", ")])
    exploded_track_df = df.explode("genres")

    return exploded_track_df, filtered_df

def find_highest_duplicate(arr):
    counts = Counter(arr)
    duplicates = [item for item, count in counts.items() if count > 1]
    if duplicates:
        return max(duplicates)
    else:
        return None
    
def get_artist_genre(highest_dup):
    first_token = highest_dup.split(',')
    if len(first_token) > 1:
        highest_dup = str(highest_dup)[1:-1]
        highest_dup = highest_dup.split(',')[0]
        highest_dup = str(highest_dup)[1:-1]
    else:
        highest_dup = str(highest_dup)[2:-2]
    return highest_dup;


# Define function to return Spotify URIs and audio feature values of top neighbors (ascending)
def n_neighbors_uri_audio(exploded_track_df, filtered_df, artist_select, genre, start_year, end_year, test_feat):
    # The artist given
    if(len(artist_select) > 0):
        print('The artist given: ', artist_select)
        print('The found in csv files: ')
        genre = find_highest_duplicate(filtered_df.genres)
        #genre_new = str(genre_new)[2:-2]
        #get the genre string value
        genre = get_artist_genre(genre)
        print('The genre of the given artist: ', genre)
        print('The start year: ', min(filtered_df.release_year))
        print('The end year: ', max(filtered_df.release_year))
        #test_feat = [acousticness, danceability, energy, instrumentalness, valence, tempo]
        print('The acousticness: ', min(filtered_df.acousticness))
        print('The acousticness: ', max(filtered_df.acousticness))
        print('The danceability: ', min(filtered_df.danceability))
        print('The danceability: ', max(filtered_df.danceability))
        print('The energy: ', min(filtered_df.energy))
        print('The energy: ', max(filtered_df.energy))
        print('The instrumentalness: ', min(filtered_df.instrumentalness))
        print('The instrumentalness: ', max(filtered_df.instrumentalness))
        print('The valence: ', min(filtered_df.valence))
        print('The valence: ', max(filtered_df.valence))
        print('The tempo: ', min(filtered_df.tempo))
        print('The tempo: ', max(filtered_df.tempo))

    #print('nnnnn ', filtered_df.release_year)
    if(len(artist_select) == 0):
        genre_data = exploded_track_df[(exploded_track_df["genres"]==genre) & (exploded_track_df["release_year"]>=start_year) & (exploded_track_df["release_year"]<=end_year)]
    else:
        genre_data = exploded_track_df[(exploded_track_df["genres"]==genre) & (exploded_track_df["release_year"]>=min(filtered_df.release_year)) & (exploded_track_df["release_year"]<=max(filtered_df.release_year))]
        #calculate test_feat from given attributes 
        #get the max and and min for the artist and get the difference 
        acousticness = max(filtered_df.acousticness) - min(filtered_df.acousticness)
        danceability = max(filtered_df.danceability) - min(filtered_df.danceability)
        energy = max(filtered_df.energy) - min(filtered_df.energy)
        instrumentalness = max(filtered_df.instrumentalness) - min(filtered_df.instrumentalness)
        valence = max(filtered_df.valence) - min(filtered_df.valence)
        tempo = max(filtered_df.tempo) - min(filtered_df.tempo)
        test_feat = [acousticness, danceability, energy, instrumentalness, valence, tempo]
    genre_data = genre_data.sort_values(by='popularity', ascending=False)[:500] # use only top 500 most popular songs
    print(len(genre_data))
    neigh = NearestNeighbors()
    neigh.fit(genre_data[audio_feats].to_numpy())
    
    n_neighbors = neigh.kneighbors([test_feat], n_neighbors=len(genre_data), return_distance=False)[0]
    
    #Search nearest neighbor
    #artists_name_lower = genre_data.iloc[n_neighbors]['artists_name_lower'].to_numpy()
    artists_id = genre_data.iloc[n_neighbors]['artists_id'].to_numpy()    
    artists_name = genre_data.iloc[n_neighbors]['artists_name'].to_numpy()
    artist_info = [genre_data.iloc[n_neighbors]['artists_id'].to_numpy(), genre_data.iloc[n_neighbors]['artists_name'].to_numpy()]
    uris = genre_data.iloc[n_neighbors]["uri"].tolist()
    audios = genre_data.iloc[n_neighbors][audio_feats].to_numpy()
    print('Artists set before removing the given artist from the list: ', len(artists_id), len(artists_name))

    if(len(artist_select) > 0):     
        indices_to_remove = np.where(artists_name == artist_select)
        #print(indices_to_remove)
        uris = np.delete(uris, indices_to_remove)
        audios = np.delete(audios, indices_to_remove)
        artists_id = np.delete(artists_id, indices_to_remove)
        artists_name = np.delete(artists_name, indices_to_remove)
        artist_info = np.delete(artist_info, indices_to_remove)
        print('After remove the entries from the list: ', len(artists_id), len(artists_name))


    return genre, genre_data, uris, audios, artists_id, artists_name, artist_info

name = 'Simon & Garfunkel'
name = 'Michael Jackson'
exploded_track_df, filtered_df = load_data(name)

genre = ''
test_feat = [0.5, 0.5, 0.5, 0.0, 0.45, 118.0]
start_year = 1960
end_year = 2000
genre, genre_data, uris, audios, artists_id, artists_name, artist_info = n_neighbors_uri_audio(exploded_track_df, filtered_df, name, genre, start_year, end_year, test_feat)


def dataset_parquet_file(genre_data):
    temp_folder_path = '/Users/dineshk/work/MLOps/spotify_dataset/parquet_data'
    
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
dataset_parquet_file(genre_data)

def create_artist_recommend(genre_data):
    temp_folder_path = '/Users/dineshk/work/MLOps/spotify_dataset/parquet_data'
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
    
create_artist_recommend(genre_data)

print("GENRE ", genre)

def find_elbowpoint():
    y_var = [genre_data.acousticness, genre_data.danceability, genre_data.energy, genre_data.instrumentalness, genre_data.valence, genre_data.tempo]
   
    for item1, item2 in zip(audio_feats, y_var):
        x = genre_data.popularity
        y = item2
        X = np.array(list(zip(x, y))).reshape(len(x), 2)
        plt.title(f'Recommended Songs for {genre} genre')
        plt.xlabel('Popularity')
        plt.ylabel(item1)
        plt.scatter(x, y)
        plt.show()

    return X;

#The Elbow Point: Optimal k Value
#find optimal k value for popularity vs loudness'
X = find_elbowpoint()

distortions = []
inertias = []
mapping1 = {}
mapping2 = {}
K = range(1, 10)

def cluster_model():
    for k in K:
        kmeanModel = KMeans(n_clusters=k, random_state=42).fit(X)
        
        distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'Minkowski'), axis=1)**2) / X.shape[0])
        inertias.append(kmeanModel.inertia_)
        
        mapping1[k] = distortions[-1]
        mapping2[k] = inertias[-1]


#Building the Clustering Model and Calculating Distortion and Inertia
#fit the K-means model for different values of k (number of clusters) and 
#calculate both the distortion and inertia for each value.
cluster_model()

def distoted_values():
    print("Distortion values:")
    for key, val in mapping1.items():
        print(f'{key} : {val}')

    plt.plot(K, distortions, 'bx-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method using Distortion')
    plt.show()

#Displaying Distortion Values
distoted_values()


def inertia_values():    
    k_range = range(1, 5)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
        y_kmeans = kmeans.fit_predict(X)
        
        plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis', marker='o', edgecolor='k', s=100)
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                    s=300, c='red', label='Centroids', edgecolor='k')
        plt.title(f'K-means Clustering (k={k})')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.grid()
        plt.show()

#Displaying Inertia Values
inertia_values()

#Reference:
#https://www.geeksforgeeks.org/machine-learning/elbow-method-for-optimal-value-of-k-in-kmeans/