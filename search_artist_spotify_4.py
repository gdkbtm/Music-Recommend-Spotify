# Import libraries
import streamlit as st
st.set_page_config(page_title="Song Recommendation", layout="wide")

import glob
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import streamlit.components.v1 as components
from numpy.random import default_rng as rng
from collections import Counter
import numpy as np

# Define function to load & expand data so that each row contains 1 genre of each track
@st.cache_data()
def load_data(name):
    filtered_df = []
    all_files = glob.glob("csv_data/*.csv")
    list_contact_csv = []
    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        #add all csv data
        list_contact_csv.append(df) 

    df = pd.concat(list_contact_csv, axis=0, ignore_index=True)
    #count total rows
    row_count = len(df)
    print(row_count)
    #remove duplicates based on uri value
    df = df.drop_duplicates(subset=['uri'], keep='first')
    #add new column
    if(len(name) > 0):
        df['artists_name_lower'] = df['artists_name'].str.lower()
        #df['artists_name'] = df['artists_name'].str.lower()
        row_count = len(df)
        print(row_count)
    
        filtered_df = df[df['artists_name_lower'] == name.lower()]
        #print(list_artist)
        #print('filtered_df length', len(filtered_df))
    df['genres'] = df.genres.apply(lambda x: [i[1:-1] for i in str(x)[1:-1].split(", ")])
    exploded_track_df = df.explode("genres")

    return exploded_track_df, filtered_df

# Define list of genres and audio features of songs for audience to choose from
genre_names = ['Dance Pop', 'Electronic', 'Electropop', 'Hip Hop', 'Jazz', 'K-pop', 'Latin', 
               'Pop', 'Pop Rap', 'R&B', 'Rock', 'Folk Rock']
audio_feats = ["acousticness", "danceability", "energy", "instrumentalness", "valence", "tempo"]

# Load data
#exploded_track_df = load_data()

def find_highest_duplicate(arr):
    counts = Counter(arr)
    duplicates = [item for item, count in counts.items() if count > 1]
    if duplicates:
        return max(duplicates)
    else:
        return None

                           #############################################

def get_artist_genre(highest_dup):
    first_token = highest_dup.split(',')
    #print(len(first_token))
    if len(first_token) > 1:
        highest_dup = str(highest_dup)[1:-1]
        highest_dup = highest_dup.split(',')[0]
        highest_dup = str(highest_dup)[1:-1]
    else:
        highest_dup = str(highest_dup)[2:-2]
    return highest_dup;
                        
## Build KNN model to retrieve top songs that are 
## closest in distance with the set of feature inputs selected by users

# Define function to return Spotify URIs and audio feature values of top neighbors (ascending)
def n_neighbors_uri_audio(exploded_track_df, filtered_df, artist_select, genre, start_year, end_year, test_feat):
    genre = genre.lower()
    # The artist given
    if(len(artist_select) > 0):
        print('The artist given: ', artist_select)
        print('The found in csv files: ')
        #print(filtered_df)
        #print('GENRES ', filtered_df.genres)
        genre = find_highest_duplicate(filtered_df.genres)
        #genre_new = str(genre_new)[2:-2]
        #get the genre string value
        genre = get_artist_genre(genre)
        print('The genre of the given artist: ', genre)
        print('The start year: ', min(filtered_df.release_year))
        print('The end year: ', max(filtered_df.release_year))

    #print('nnnnn ', filtered_df.release_year)
    if(len(artist_select) == 0):
        genre_data = exploded_track_df[(exploded_track_df["genres"]==genre) & (exploded_track_df["release_year"]>=start_year) & (exploded_track_df["release_year"]<=end_year)]
    else:
        genre_data = exploded_track_df[(exploded_track_df["genres"]==genre) & (exploded_track_df["release_year"]>=min(filtered_df.release_year)) & (exploded_track_df["release_year"]<=max(filtered_df.release_year))]
    #print(len(genre_data))
    genre_data = genre_data.sort_values(by='popularity', ascending=False)[:500] # use only top 500 most popular songs
    
    neigh = NearestNeighbors()
    neigh.fit(genre_data[audio_feats].to_numpy())
    
    n_neighbors = neigh.kneighbors([test_feat], n_neighbors=len(genre_data), return_distance=False)[0]
    
    #artists_id = exploded_track_df['artists_id']
    artists_id = genre_data.iloc[n_neighbors]['artists_id'].to_numpy()
    artists_name_lower = genre_data.iloc[n_neighbors]['artists_name_lower'].to_numpy()
    artists_name = genre_data.iloc[n_neighbors]['artists_name'].to_numpy()
    artist_info = [genre_data.iloc[n_neighbors]['artists_id'].to_numpy(), genre_data.iloc[n_neighbors]['artists_name'].to_numpy()]
    uris = genre_data.iloc[n_neighbors]["uri"].tolist()
    audios = genre_data.iloc[n_neighbors][audio_feats].to_numpy()
    print('Artists set before removing the given artist from the list: ', len(artists_id), len(artists_name))
    #print(artists_name_lower)

    indices_to_remove = np.where(artists_name_lower == artist_select)
    uris = np.delete(uris, indices_to_remove)
    audios = np.delete(audios, indices_to_remove)
    artists_id = np.delete(artists_id, indices_to_remove)
    artists_name = np.delete(artists_name, indices_to_remove)
    artist_info = np.delete(artist_info, indices_to_remove)

    print('After remove ', len(artists_id), len(artists_name))
    return uris, audios, artists_id, artists_name, artist_info


                           #############################################

## Build frontend app layout - a dashboard that allows users to customize songs they want to listen to

# Design dashboard layout with customizeable sliders

def main():
    name = str()
    form = []
    st.title("Personalized Song Recommendations")
    
    st.sidebar.title("Music Recommender App")
    st.sidebar.header("Welcome!")
    st.sidebar.markdown("Discover your soon-to-be favorite songs by selecting genres and audio features.")
    st.sidebar.markdown("Tips: Play around with different settings and listen to song previews to test the system!")    
    
    # Add buttons to the sidebar
    if st.sidebar.button("Check out my other projects"):
        st.sidebar.markdown("[https://hahoangpro.wixsite.com/datascience]")
    if st.sidebar.button("Connect with me on LinkedIn"):
        st.sidebar.markdown("[https://www.linkedin.com/in/ha-hoang-86a80814a/]")
    #form = st.form(key='my_form', clear_on_submit=True)
    form = st.form(key='my_form')
    name = form.text_input(label='Enter Artist Name')
    submit_button = form.form_submit_button(label='Submit')
    #print('aaaaa ', name)
    if(len(name) > 0):
        # Load data    
        exploded_track_df, filtered_df = load_data(name)    
        #print('filtered_df length', len(filtered_df))
        if(len(filtered_df) == 0):
            return 0   
        call_container(exploded_track_df, filtered_df, name)
    else:
        exploded_track_df, filtered_df = load_data(name)    
        #print('XXXXX ' , filtered_df)
        call_container(exploded_track_df, filtered_df, name)

def call_container(exploded_track_df, filtered_df, name):
    with st.container():
        col1, col2, col3, col4 = st.columns((2,0.5,1,0.5))
        with col3:
            st.markdown("***Select genre:***")
            genre = st.radio(
                "",
                genre_names, index=genre_names.index("Rock"))
        with col1:
            st.markdown("***Select features to customize:***")
            start_year, end_year = st.slider('Select year range', 1960, 2019, (1960, 1985))
            acousticness = st.slider('Acousticness', 0.0, 1.0, 0.5)
            danceability = st.slider('Danceability', 0.0, 1.0, 0.5)
            energy = st.slider('Energy', 0.0, 1.0, 0.5)
            valence = st.slider('Positiveness (Valence)', 0.0, 1.0, 0.45)
            instrumentalness = st.slider('Instrumentalness', 0.0, 1.0, 0.0)
            tempo = st.slider('Tempo', 0.0, 244.0, 118.0)
    
    ## Display 6 top songs (closest neighbors) to recommend based on selected features
    tracks_per_page = 9
    test_feat = [acousticness, danceability, energy, instrumentalness, valence, tempo]
    uris, audios, artists_id, artists_name, artist_info = n_neighbors_uri_audio(exploded_track_df, filtered_df, name, genre, start_year, end_year, test_feat)
    #print(artist_info)
    # Use Spotify Developer Widget to display iframe with classic HTML
    tracks = []
    artists = []
    artist_ids = []
    artists_info = []
    #print(len(uris))
    #print(len(artists_name))
    for uri in uris:
        #print(uri)
        track = """<iframe src="https://open.spotify.com/embed/track/{}" width="260" height="380" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>""".format(uri)
        artist = artists_name
        id = artists_id
        info = artist_info
        tracks.append(track) 
        artists.append(artist)
        artist_ids.append(id)
        artists_info.append(info)
        #print(audios)

    # Add "Recommend More Songs" button to have more options
    
    ## Use Streamlit's session_state to check if users alter any inputs between sessions
    ## If users alter any inputs, the recommendation starts from the 1st track of top 'neighbors'
    ## If users continue to press "Recommend More Songs" without changing inputs, the top neighbors 
    ## will be traversed till the end of the top neighbors list

    current_inputs = [genre, start_year, end_year] + test_feat
    
    try: 
        previous_inputs = st.session_state['previous_inputs']
    except KeyError:
        previous_inputs = None
        
    if current_inputs != previous_inputs:
        st.session_state['start_track_i'] = 0
        st.session_state['previous_inputs'] = current_inputs  
    
    ## Design layout of the "Recommend More Songs" button    
    # Initialize start_track_i if not present in session state
    if 'start_track_i' not in st.session_state:
        st.session_state['start_track_i'] = 0
        st.write("start_track_i initialized:", st.session_state['start_track_i'])

    # Add "Recommend More Songs" button
    if st.button("Recommend More Songs"):
        if st.session_state['start_track_i'] < len(tracks):
            st.session_state['start_track_i'] += tracks_per_page  # Show 6 more songs          

    with st.container():
        col1, col2, col3 = st.columns(3)  # Create 3 columns for a 3x3 grid

    current_tracks = tracks[st.session_state['start_track_i']: st.session_state['start_track_i'] + tracks_per_page]
    current_audios = audios[st.session_state['start_track_i']: st.session_state['start_track_i'] + tracks_per_page]
    current_artists = artists[st.session_state['start_track_i']: st.session_state['start_track_i'] + tracks_per_page]
    current_artist_ids = artist_ids[st.session_state['start_track_i']: st.session_state['start_track_i'] + tracks_per_page]
    current_artist_info = artists_info[st.session_state['start_track_i']: st.session_state['start_track_i'] + tracks_per_page]
    #print(info[0], info[1])
    #print(track)
    #print(current_artist_ids)
    list_artists = list(dict.fromkeys(artist))
    list_artist_ids = list(dict.fromkeys(id))
    list_artist_info = list(dict.fromkeys(info[0])), list(dict.fromkeys(info[1]))
    #print(list_artist_info)
    l_artists = []
    l_artists_id = []
    for i, (track, audio, artist, artist_id) in enumerate(zip(current_tracks, current_audios, current_artists, current_artist_ids)):
        #print(track, artist[i], artist_id[i])
        l_artists.append(artist[i])
        l_artists_id.append(artist_id[i])

        #print(list(set(artist)))
        if i % 3 == 0:
            #print(track)
            #print(list_artists)
            with col1:
                components.html(
                    track, 
                    height=400, 
                )
            
        elif i % 3 == 1:
            #print(artist)
            with col2:
                components.html(
                    track,
                    height=400,
                )
        else:
            with col3:
                #print(artist)
                components.html(
                    track,
                    height=400,
                )
    #list_artists  
    #list_artist_ids 
    #artist_info = dict(zip(list_artists, list_artist_ids))
    #artist_info 
    
    list_artists = list(dict.fromkeys(l_artists))
    list_artist_ids = list(dict.fromkeys(l_artists_id))
    #print(list_artists)
    #print(list_artist_ids)
#    artist_info = dict(zip(list_artists, list_artist_ids))
#    artist_info_2 = (list_artists, list_artist_ids)
#    print(artist_info_2[0])

#    with st.container():
#        with col1:
#            for key, value in artist_info.items():
#                hyperlink_format = """<a href="https://open.spotify.com/artist/{}" width="260" height="20" target="_blank">{}</a>""".format(value, key)
#                components.html(hyperlink_format)

    heading =  'Search Artist Info'
    st.write(heading)
    
    #combined_list = list(zip(list_artists, list_artist_ids))
    combined_list = [[list_artists[i], list_artist_ids[i]] for i in range(len(list_artists))]

    #print(combined_list)

    for j in combined_list:
        #j[1] = """<a href="https://open.spotify.com/artist/{}" width="260" height="20" target="_blank">{}</a>""".format(j[1], j[0]) 
        j[1] = 'https://open.spotify.com/artist/{}'.format(j[1])
        print(j[1])
    
    # Create a dynamic array of headers
    headers = ["Artist", "Check info @ Spotify"]

    # Create a DataFrame using the dynamic data
    df = pd.DataFrame(combined_list, columns=headers)
    
    st.table(df)

    if st.session_state['start_track_i'] >= len(tracks):
        st.write("No more songs to recommend")
        
    #st.session_state.reset_form = True
    #st.rerun() # Rerun the script to apply the reset            
if __name__ == "__main__":
    message = main()
    if(message == 0):
        print("No artist found")
        st.write('No artist found for the criteria. Please try')
# Note: Install streamlit by typing in the command line "conda install -c conda-forge streamlit" to install latest version
# Test if streamlit is operable by typing "streamlit hello" 

