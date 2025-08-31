# Import libraries
import streamlit as st
st.set_page_config(page_title="Song Recommendation", layout="wide")

import glob
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import streamlit.components.v1 as components
from numpy.random import default_rng as rng

# Define function to load & expand data so that each row contains 1 genre of each track
@st.cache_data()
def load_data():
    all_files = glob.glob("csv_data/*.csv")
    #df = pd.read_csv("data/filtered_track_df.csv")
    #df = pd.read_csv("data/filtered_track_classic_df.csv")

    li = []

    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        li.append(df)

    df = pd.concat(li, axis=0, ignore_index=True)
    #count total rows
    row_count = len(df)
    print('ZZZZZZZ ', row_count)
    #remove duplicates based on uri value
    df = df.drop_duplicates(subset=['uri'], keep='first')
    row_count = len(df)
    print(row_count)
    #df = pd.read_csv(filenames)
    df['genres'] = df.genres.apply(lambda x: [i[1:-1] for i in str(x)[1:-1].split(", ")])
    exploded_track_df = df.explode("genres")
    return exploded_track_df

# Define list of genres and audio features of songs for audience to choose from
genre_names = ['Dance Pop', 'Electronic', 'Electropop', 'Hip Hop', 'Jazz', 'K-pop', 'Latin', 
               'Pop', 'Pop Rap', 'R&B', 'Rock', 'Folk Rock']
audio_feats = ["acousticness", "danceability", "energy", "instrumentalness", "valence", "tempo"]

# Load data
exploded_track_df = load_data()

                           #############################################
                           
## Build KNN model to retrieve top songs that are 
## closest in distance with the set of feature inputs selected by users


# Define function to return Spotify URIs and audio feature values of top neighbors (ascending)
def n_neighbors_uri_audio(genre, start_year, end_year, test_feat):
    genre = genre.lower()
    #print(exploded_track_df["genres"])
    #print(exploded_track_df['artists_id'], exploded_track_df['artists_name'])
    genre_data = exploded_track_df[(exploded_track_df["genres"]==genre) & (exploded_track_df["release_year"]>=start_year) & (exploded_track_df["release_year"]<=end_year)]
    print(len(genre_data))
    genre_data = genre_data.sort_values(by='popularity', ascending=False)[:500] # use only top 500 most popular songs
    
    neigh = NearestNeighbors()
    neigh.fit(genre_data[audio_feats].to_numpy())
    
    n_neighbors = neigh.kneighbors([test_feat], n_neighbors=len(genre_data), return_distance=False)[0]
    
    #artists_id = exploded_track_df['artists_id']
    artists_id = genre_data.iloc[n_neighbors]['artists_id'].to_numpy()
    artists_name = genre_data.iloc[n_neighbors]['artists_name'].to_numpy()
    artist_info = [genre_data.iloc[n_neighbors]['artists_id'].to_numpy(), genre_data.iloc[n_neighbors]['artists_name'].to_numpy()]
    uris = genre_data.iloc[n_neighbors]["uri"].tolist()
    audios = genre_data.iloc[n_neighbors][audio_feats].to_numpy()
    #print(artists_name[0])
    #print(artist_info)
    return uris, audios, artists_id, artists_name, artist_info


                           #############################################

## Build frontend app layout - a dashboard that allows users to customize songs they want to listen to

# Design dashboard layout with customizeable sliders

def main():
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
    uris, audios, artists_id, artists_name, artist_info = n_neighbors_uri_audio(genre, start_year, end_year, test_feat)
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
        print('button clicked')
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

    print(combined_list)

    for j in combined_list:
        #j[1] = """<a href="https://open.spotify.com/artist/{}" width="260" height="20" target="_blank">{}</a>""".format(j[1], j[0]) 
        j[1] = 'https://open.spotify.com/artist/{}'.format(j[1])
        print(j[1])
    
    # Create a dynamic array of headers
    headers = ["Artist", "Check info @ Spotify"]

    # Create a DataFrame using the dynamic data
    df = pd.DataFrame(combined_list, columns=headers)
    
    st.table(df)

    data = {
    'Name': ['John', 'Alice', 'Bob'],
    'Website': ['https://www.example.com',
                'https://www.google.com',
                'https://www.openai.com']
    }
    # Create the DataFrame
    df = pd.DataFrame(data)
    #st.dataframe(df, column_config={"Website": st.column_config.LinkColumn()})

    if st.session_state['start_track_i'] >= len(tracks):
        st.write("No more songs to recommend")
        
            
if __name__ == "__main__":
    main()

# Note: Install streamlit by typing in the command line "conda install -c conda-forge streamlit" to install latest version
# Test if streamlit is operable by typing "streamlit hello" 

