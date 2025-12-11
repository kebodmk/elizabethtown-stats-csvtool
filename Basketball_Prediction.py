# ===========================
#        IMPORTS
# ===========================

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pickle
import io
import os
import warnings

# Machine Learning
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

warnings.filterwarnings("ignore") 

@st.cache_resource
def load_model():    
    player_data_list = []
    team_data_list = []
    
    directory = "./player_game_stats/"
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            filepath = os.path.join(directory, filename)
            df = pd.read_csv(filepath)
            player_data_list.append(df)
    print(f"Loaded {len(player_data_list)} player data csv files.")
    
    directory = "./team_game_stats/"
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            filepath = os.path.join(directory, filename)
            df = pd.read_csv(filepath)
            team_data_list.append(df)
    print(f"Loaded {len(team_data_list)} team data csv files.")
    
    player_stats = pd.concat(player_data_list, axis=0)
    team_stats = pd.concat(team_data_list, axis=0)
    
    
    
    player_stats.drop(columns=['GS'], inplace=True)
    team_stats.drop(columns=['Attendance','Referees','Notes'], inplace=True) # Null columns
    
    def reformatted_names_mask(name):
        name = name.strip()
        if ',' in name:
            l_f_names = [n.strip() for n in name.split(',')]
            name = l_f_names[1]+' '+l_f_names[0]
        return name
    
    player_stats['Player'] = player_stats['Player'].apply(reformatted_names_mask)
    
    
    
    team_mapping = {
        'Albright College': 'albright',
        'Moravian University': 'moravian',
        'Albright': 'albright',
        'Goucher': 'goucher',
        'Moravian': 'moravian',
        'Scranton': 'scranton',
        'Wilkes': 'wilkes',
        'Alvernia': 'alvernia',
        'Elizabethtown': 'elizabethtown',
        'Juniata': 'juniata',
        'Arcadia': 'arcadia',
        'Babson': 'babson',
        'Bates': 'bates',
        'Bethany': 'bethany',
        'Bethel (Minn.)': 'bethel (minn)',
        'Brooklyn': 'brooklyn',
        'Drew': 'drew',
        'Bryn Mawr': 'bryn mawr',
        'Cabrini': 'cabrini',
        'Capital': 'capital',
        'Catholic': 'catholic',
        'Bridgewater (VA)': 'bridgewater (va)',
        'Eastern': 'eastern',
        'Gallaudet': 'gallaudet',
        'Ithaca': 'ithaca',
        'Lycoming': 'lycoming',
        'Marymount (VA)': 'marymount (va)',
        'McDaniel': 'mcdaniel',
        'Randolph-Macon': 'randolph-macon',
        'Salisbury': 'salisbury',
        "St. Mary's (MD)": 'st marys (md)',
        'Susquehanna': 'susquehanna',
        'Trinity Washington': 'trinity washington',
        'Vassar': 'vassar',
        'York (Pa.)': 'york',
        'CCNY': 'ccny',
        'Chatham': 'chatham',
        'Chris. Newport': 'chris newport',
        'Clarks Summit': 'clarks summit',
        'Colorado Col.': 'colorado col',
        'Cortland': 'cortland',
        'Delaware Valley': 'delaware valley',
        'DeSales': 'desales',
        'Dickinson': 'dickinson',
        'Drew University': 'drew',
        'FDU-Florham': 'fdu-florham',
        'JOHNJAYW': 'john jay',
        'DREW': 'drew',
        "Juniata Women's Basketball": 'juniata',
        'Kean': 'kean',
        'Medgar Evers College': 'medgar evers',
        'Rhodes': 'rhodes',
        'Rutgers University-Camden': 'rutgers-camden',
        'Wm. Paterson': 'wm paterson',
        'Yeshiva': 'yeshiva',
        'East Texas Baptist': 'east texas baptist',
        'Elizabethtown College': 'elizabethtown',
        'Franklin & Marshall': 'franklin and marshall',
        'John Carroll': 'john carroll',
        "King's (PA)": 'kings (pa)',
        'Lebanon Valley': 'lebanon valley',
        'Roanoke': 'roanoke',
        'Rowan': 'rowan',
        'Stevenson': 'stevenson',
        'Framingham St.': 'framingham st',
        'George Fox': 'george fox',
        'Goucher College': 'goucher',
        'Alfred': 'alfred',
        'Hunter': 'hunter',
        'Neumann': 'neumann',
        'Notre Dame (MD)': 'notre dame (md)',
        'Penn St.-Berks': 'penn state-berks',
        'Shenandoah': 'shenandoah',
        'Trinity (D.C.)': 'trinity (dc)',
        'Gwynedd Mercy Univ.': 'gwynedd mercy',
        'Gwynedd Mercy': 'gwynedd mercy',
        'Haverford': 'haverford',
        'Hiram': 'hiram',
        'Immaculata University': 'immaculata',
        'Immaculata': 'immaculata',
        'John Jay': 'john jay',
        'Johns Hopkins': 'johns hopkins',
        'Juniata College': 'juniata',
        'Moravian College': 'moravian',
        'Bethany (WV)': 'bethany (wv)',
        'Elmira': 'elmira',
        'Grove City': 'grove city',
        'Hood': 'hood',
        'Lehman': 'lehman',
        'Marywood': 'marywood',
        'Misericordia': 'misericordia',
        'Penn State-Harrisburg': 'penn state-harrisburg',
        'Pitt.-Bradford': 'pitt-bradford',
        'Saint Vincent': 'st vincent',
        'Wash. & Lee': 'wash and lee',
        'Keystone': 'keystone',
        "King's (Pa.)": 'kings (pa)',
        'Lancaster Bible': 'lancaster bible',
        'Lebanon Valley College': 'lebanon valley',
        'Loras': 'loras',
        'Bryn Athyn': 'bryn athyn',
        'LeTourneau': 'letourneau',
        'Nazareth': 'nazareth',
        'Penn College': 'penn college',
        'Penn St.-Altoona': 'penn state-altoona',
        'LYCOW': 'lycoming',
        'Lynchburg': 'lynchburg',
        'Mary Washington': 'mary washington',
        'Maryville (TN)': 'maryville (tn)',
        'Messiah': 'messiah',
        'Montclair St.': 'montclair st',
        'Lebanon Valley Col.': 'lebanon valley',
        'Muhlenberg College': 'muhlenberg',
        'Susquehanna University': 'susquehanna',
        'TCNJ': 'tcnj',
        'York (Pa.) College': 'york',
        'Emmanuel (MA)': 'emmanuel (ma)',
        'MORAVIAN': 'moravian',
        'Muhlenberg': 'muhlenberg',
        'Russell Sage College': 'russell sage',
        'Mount Aloysius': 'mount aloysius',
        'Mt. St. Mary (NY)': 'mt st mary (ny)',
        'Muskingum': 'muskingum',
        'New Jersey City': 'new jersey city',
        'Oberlin': 'oberlin',
        'Ohio Northern': 'ohio northern',
        'Ohio Wesleyan': 'ohio wesleyan',
        'Penn St.-Behrend': 'penn state-behrend',
        'Penn St.-Lehigh Val.': 'penn state-lehigh val',
        'Penn State-Altoona': 'penn state-altoona',
        'PSHW': 'penn state-harrisburg',
        'Raritan Valley CC': 'raritan valley',
        'Raritan Valley': 'raritan valley',
        'Rhode Island Col.': 'rhode island col',
        'RIT': 'rit',
        'Rosemont': 'rosemont',
        'Russell Sage': 'russell sage',
        'Rutgers-Camden': 'rutgers-camden',
        'Saint Elizabeth': 'st elizabeth',
        'Colby': 'colby',
        'NYU': 'nyu',
        'Stevens': 'stevens',
        'Tufts': 'tufts',
        'Wartburg': 'wartburg',
        'St. Joseph (Conn.)': 'st joseph (conn)',
        "St. Joseph's (Brkln)": 'st josephs (brkln)',
        "St. Joseph's (ME)": 'st josephs (me)',
        'St. Vincent': 'st vincent',
        'Stevens Institute': 'stevens',
        'Stockton': 'stockton',
        'SUNY Brockport': 'suny brockport',
        'SUNY Geneseo': 'suny geneseo',
        'Concordia Wisconsin': 'concordia wisconsin',
        'Gettysburg': 'gettysburg',
        'St. John Fisher': 'st john fisher',
        'Wittenberg': 'wittenberg',
        'WPI': 'wpi',
        'Univ. of Scranton': 'scranton',
        'University of Scranton': 'scranton',
        'Ursinus': 'ursinus',
        'Valley Forge': 'valley forge',
        'Wesleyan (CT)': 'wesleyan (ct)',
        'Western New Eng.': 'western new eng',
        'Widener': 'widener',
        'Wilson': 'wilson',
        'Wis.-Stout': 'wis-stout',
        'Wis.-Whitewater': 'wis-whitewater',
        'Wittenberg Univ.': 'wittenberg',
        'York (PA)': 'york',
    }
    
    team_stats['Team'] = team_stats['Team'].apply(lambda team: team_mapping[team])
    player_stats['Team'] = player_stats['Team'].apply(lambda team: team_mapping[team])
    
    
    
    team_stats['FG%'] = team_stats['FG'].apply(lambda x: int(x.split('-')[0]) / int(x.split('-')[1]) if int(x.split('-')[1])!=0 else 0)
    team_stats['3PT%'] = team_stats['3PT'].apply(lambda x: int(x.split('-')[0]) / int(x.split('-')[1]) if int(x.split('-')[1])!=0 else 0)
    team_stats['FT%'] = team_stats['FT'].apply(lambda x: int(x.split('-')[0]) / int(x.split('-')[1]) if int(x.split('-')[1])!=0 else 0)
    team_stats['ORB'] = team_stats['ORB-DRB'].apply(lambda x: int(x.split('-')[0]))
    team_stats['DRB'] = team_stats['ORB-DRB'].apply(lambda x: int(x.split('-')[1]))
    
    team_stats.drop(columns=['FG', '3PT', 'FT', 'ORB-DRB'], inplace=True)
    
    player_stats['FG%'] = player_stats['FG'].apply(lambda x: int(x.split('-')[0]) / int(x.split('-')[1]) if int(x.split('-')[1])!=0 else 0)
    player_stats['3PT%'] = player_stats['3PT'].apply(lambda x: int(x.split('-')[0]) / int(x.split('-')[1]) if int(x.split('-')[1])!=0 else 0)
    player_stats['FT%'] = player_stats['FT'].apply(lambda x: int(x.split('-')[0]) / int(x.split('-')[1]) if int(x.split('-')[1])!=0 else 0)
    player_stats['ORB'] = player_stats['ORB-DRB'].apply(lambda x: int(x.split('-')[0]))
    player_stats['DRB'] = player_stats['ORB-DRB'].apply(lambda x: int(x.split('-')[1]))
    
    player_stats.drop(columns=['FG', '3PT', 'FT', 'ORB-DRB'], inplace=True)
    
    
    
    maxes = team_stats.reset_index().groupby(by='game_id')['PTS'].idxmax() # get indexes of winning matchups
    
    l = np.arange(team_stats.shape[0])
    wins_mask = [val in maxes.values for val in l] # create boolean list of whether the game is a winner
    team_stats['is_win'] = wins_mask
    
    player_stats['is_win'] = player_stats.apply(lambda series: bool(team_stats[(team_stats['game_id']==series.loc['game_id'])&(team_stats['Team']==series.loc['Team'])]['is_win'].iloc[0]), axis=1)
    
    
    
    conference_teams = ['goucher',
    'scranton',
    'elizabethtown',
    'catholic',
    'susquehanna',
    'juniata',
    'drew',
    'moravian',
    'lycoming',
    'wilkes']
    
    
    
    
    
    df = team_stats.copy()
    
    # Train on only conference teams to allow full rolling 5 game window
    # df = team_stats[team_stats['Team'].apply(lambda team: team in conference_teams)].copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by=['Team', 'Date'])
    df.reset_index(drop=True, inplace=True)
    
    # Select columns you want to compute rolling averages on
    rolling_stats_cols = ['PTS', 'REB', 'A', 'TO', 'STL', 'BLK', 'PF', 'FG%', '3PT%', 'FT%', 'ORB', 'DRB', 'is_win']
    
    # Compute rolling averages (last 5 games, excluding current game)
    df_rolling = (
        df.groupby('Team')[rolling_stats_cols]
        .rolling(window=5, min_periods=2, closed="left")
        .mean()
        .reset_index()
    )
    
    # Merge back into original df
    for col in rolling_stats_cols:
        df[f'{col}_rolling_5'] = df_rolling[col]
    df = df.dropna(axis=0)
    
    # Create copies of dataframes for usage
    df_team = team_stats[team_stats['Team'].apply(lambda team: team in conference_teams)].copy()
    df_player = player_stats[player_stats['Team'].apply(lambda team: team in conference_teams)].copy()
    df_team_dates = df_team[['game_id', 'Date']].drop_duplicates()
    # Join player dataframe by game_id
    df_player = df_player.merge(
        df_team_dates,
        on='game_id',
        how='left'
    )
    
    # Sort values in player table
    df_player['Date'] = pd.to_datetime(df_player['Date'])
    df_player = df_player.sort_values(by=['Player', 'Date'])
    
    player_rolling_cols = ['PTS', 'REB', 'A', 'TO', 'STL', 'BLK', 'PF',
                           'FG%', '3PT%', 'FT%', 'ORB', 'DRB']
    # Calculate last-5-game rolling averages for individual players
    player_rolling = (
        df_player.groupby('Player')[player_rolling_cols]
        .rolling(window=5, min_periods=3, closed="left")    # <= EXCLUDES current game
        .mean()
        .reset_index()
    )
    
    # Rename columns to allow for player & team data differentiation
    df_player.reset_index(drop=True, inplace=True)
    for col in df_player.columns:
        if col not in ['game_id', 'Team', 'Date']:
            df_player.rename(columns={col: f'{col}_player'}, inplace=True)
    # Copy over rolling averages to respective player
    for col in player_rolling_cols:
        df_player[f'{col}_rolling_5_player_max'] = player_rolling[col]
    # Fetch player with max rolling average per game, per team
    df_player.dropna(axis=0, inplace=True)
    players_max_rollings = df_player.groupby(['game_id','Team','Date']).max()
    
    # Join player rolling averages back to main dataframe by game
    df = df.merge(
        players_max_rollings,
        on=['game_id', 'Team', 'Date'],
        how='left'
    )
    
    
    # Split dataframe into two views: one per game per team
    team_a = df[df['Home/Away'] == 'home'].copy()
    team_b = df[df['Home/Away'] == 'away'].copy()
    
    # Merge home team (A) and away team (B) on game_id
    matchups = team_a.merge(
        team_b,
        on=['game_id','Date'],
        suffixes=('_A', '_B')
    )
    
    feature_cols_team = [
        'PTS_rolling_5', 'REB_rolling_5', 'A_rolling_5', 'TO_rolling_5', 'STL_rolling_5',
        'BLK_rolling_5', 'PF_rolling_5', 'FG%_rolling_5', '3PT%_rolling_5', 'FT%_rolling_5',
        'ORB_rolling_5', 'DRB_rolling_5', 'is_win_rolling_5'
    ]
    feature_cols_players = [
        'PTS_rolling_5_player_max', 'REB_rolling_5_player_max', 'A_rolling_5_player_max', 'TO_rolling_5_player_max', 'STL_rolling_5_player_max',
        'BLK_rolling_5_player_max', 'PF_rolling_5_player_max', 'FG%_rolling_5_player_max', '3PT%_rolling_5_player_max', 'FT%_rolling_5_player_max',
        'ORB_rolling_5_player_max', 'DRB_rolling_5_player_max'
    ]
    
    # Retrieve final columns used for prediction
    rolling_matchups = matchups[
        ['game_id', 'Date', 'Team_A', 'Team_B', 'is_win_A'] +  # keep winner label
        [f"{c}_A" for c in feature_cols_team] +
        [f"{c}_B" for c in feature_cols_team] +
        [f"{c}_A" for c in feature_cols_players] +
        [f"{c}_B" for c in feature_cols_players]
    ].copy()
    
    # Relabel the `label` column indicating a win for the home team
    rolling_matchups.rename(columns={"is_win_A": "label"}, inplace=True)
    rolling_matchups['label'] = rolling_matchups['label'].astype(int)
    
    
    rolling_matchups.isnull().sum()
    rolling_matchups.dropna(axis=0, inplace=True)
    
    
    
    X = rolling_matchups.drop(columns=['label', 'game_id', 'Date', 'Team_A', 'Team_B'])
    y = rolling_matchups['label']
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=101)
    x_cols = list(rolling_matchups.drop(columns=['label', 'game_id', 'Date', 'Team_A', 'Team_B']).columns)
    X_train = X_train[x_cols]
    X_test = X_test[x_cols]
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    from sklearn.metrics import confusion_matrix, classification_report
    
    # Logistic Regression
    from sklearn.linear_model import LogisticRegression
    
    log = LogisticRegression(random_state=101)
    log.fit(X_train_scaled, y_train)
    
    pred_y_log = log.predict(X_test_scaled)
    
    print(confusion_matrix(y_test, pred_y_log))
    print(classification_report(y_test, pred_y_log))
    
    # KNN
    from sklearn.neighbors import KNeighborsClassifier
    
    knn = KNeighborsClassifier()
    knn.fit(X_train_scaled, y_train)
    
    pred_y_knn = knn.predict(X_test_scaled)
    
    print(confusion_matrix(y_test, pred_y_knn))
    print(classification_report(y_test, pred_y_knn))
    
    # SVM
    from sklearn.svm import SVC
    
    svc = SVC(random_state=101)
    svc.fit(X_train_scaled, y_train)
    pred_y_svc = svc.predict(X_test_scaled)
    
    
    
    df = team_stats.copy()
    
    # Train on only conference teams to allow full rolling 5 game window
    # df = team_stats[team_stats['Team'].apply(lambda team: team in conference_teams)].copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by=['Team', 'Date'])
    df.reset_index(drop=True, inplace=True)
    
    # Select columns you want to compute rolling averages on
    rolling_stats_cols = ['PTS', 'REB', 'A', 'TO', 'STL', 'BLK', 'PF', 'FG%', '3PT%', 'FT%', 'ORB', 'DRB', 'is_win']
    
    # Compute rolling averages (last 5 games, excluding current game)
    df_rolling = (
        df.groupby('Team')[rolling_stats_cols]
        .rolling(window=5, min_periods=2, closed="left")
        .mean()
        .reset_index()
    )
    
    # Merge back into original df
    for col in rolling_stats_cols:
        df[f'{col}_rolling_5'] = df_rolling[col]
    df = df.dropna(axis=0)
    
    # Create copies of dataframes for usage
    df_team = team_stats[team_stats['Team'].apply(lambda team: team in conference_teams)].copy()
    df_player = player_stats[player_stats['Team'].apply(lambda team: team in conference_teams)].copy()
    df_team_dates = df_team[['game_id', 'Date']].drop_duplicates()
    # Join player dataframe by game_id
    df_player = df_player.merge(
        df_team_dates,
        on='game_id',
        how='left'
    )
    
    # Sort values in player table
    df_player['Date'] = pd.to_datetime(df_player['Date'])
    df_player = df_player.sort_values(by=['Player', 'Date'])
    
    player_rolling_cols = ['PTS', 'REB', 'A', 'TO', 'STL', 'BLK', 'PF',
                           'FG%', '3PT%', 'FT%', 'ORB', 'DRB']
    # Calculate last-5-game rolling averages for individual players
    player_rolling = (
        df_player.groupby('Player')[player_rolling_cols]
        .rolling(window=5, min_periods=3, closed="left") # Excludes current game
        .mean()
        .reset_index()
    )
    
    # Rename columns to allow for player & team data differentiation
    df_player.reset_index(drop=True, inplace=True)
    for col in df_player.columns:
        if col not in ['game_id', 'Team', 'Date']:
            df_player.rename(columns={col: f'{col}_player'}, inplace=True)
    # Copy over rolling averages to respective player
    for col in player_rolling_cols:
        df_player[f'{col}_rolling_5_player'] = player_rolling[col]
    df_player.dropna(axis=0, inplace=True)
    
    # Fetch player with max rolling average per game, per team
    # players_max_rollings = df_player.groupby(['game_id','Team','Date']).max()
    # Custom function to get top 4 values (max and second max)
    def top4(s):
        vals = s.nlargest(4).tolist()
        # Ensure at least 4 values (or fill with NaN if missing)
        return pd.Series(vals + [float('nan')] * (4 - len(vals)), index=['max', 'second_max', 'third_max', 'fourth_max'])
    result_player_maxes = (
        df_player.groupby(['game_id','Team','Date'])[[f'{col}_rolling_5_player' for col in player_rolling_cols]]
          .apply(lambda g: g.apply(top4))
    )
    
    # Reorganize all columns into 2D DataFrame
    result_player_maxes = result_player_maxes.unstack()
    result_player_maxes.columns = [f"{col}_{suffix}" for col, suffix in result_player_maxes.columns]
    
    
    # Join player rolling averages back to main dataframe by game
    df = df.merge(
        result_player_maxes,
        on=['game_id', 'Team', 'Date'],
        how='left'
    )
    
    
    # Split dataframe into two views: one per game per team
    team_a = df[df['Home/Away'] == 'home'].copy()
    team_b = df[df['Home/Away'] == 'away'].copy()
    
    # Merge home team (A) and away team (B) on game_id
    matchups = team_a.merge(
        team_b,
        on=['game_id','Date'],
        suffixes=('_A', '_B')
    )
    
    feature_cols_team_inc = [
        'PTS_rolling_5', 'REB_rolling_5', 'A_rolling_5', 'TO_rolling_5', 'STL_rolling_5',
        'BLK_rolling_5', 'PF_rolling_5', 'FG%_rolling_5', '3PT%_rolling_5', 'FT%_rolling_5',
        'ORB_rolling_5', 'DRB_rolling_5', 'is_win_rolling_5'
    ]
    feature_cols_players_inc = [
        'PTS_rolling_5_player_max', 'REB_rolling_5_player_max', 'A_rolling_5_player_max', 'TO_rolling_5_player_max', 'STL_rolling_5_player_max',
        'BLK_rolling_5_player_max', 'PF_rolling_5_player_max', 'FG%_rolling_5_player_max', '3PT%_rolling_5_player_max', 'FT%_rolling_5_player_max',
        'ORB_rolling_5_player_max', 'DRB_rolling_5_player_max',
        'PTS_rolling_5_player_second_max', 'REB_rolling_5_player_second_max', 'A_rolling_5_player_second_max', 'TO_rolling_5_player_second_max', 'STL_rolling_5_player_second_max',
        'BLK_rolling_5_player_second_max', 'PF_rolling_5_player_second_max', 'FG%_rolling_5_player_second_max', '3PT%_rolling_5_player_second_max', 'FT%_rolling_5_player_second_max',
        'ORB_rolling_5_player_second_max', 'DRB_rolling_5_player_second_max',
        'PTS_rolling_5_player_third_max', 'REB_rolling_5_player_third_max', 'A_rolling_5_player_third_max', 'TO_rolling_5_player_third_max', 'STL_rolling_5_player_third_max',
        'BLK_rolling_5_player_third_max', 'PF_rolling_5_player_third_max', 'FG%_rolling_5_player_third_max', '3PT%_rolling_5_player_third_max', 'FT%_rolling_5_player_third_max',
        'ORB_rolling_5_player_third_max', 'DRB_rolling_5_player_third_max',
        'PTS_rolling_5_player_fourth_max', 'REB_rolling_5_player_fourth_max', 'A_rolling_5_player_fourth_max', 'TO_rolling_5_player_fourth_max', 'STL_rolling_5_player_fourth_max',
        'BLK_rolling_5_player_fourth_max', 'PF_rolling_5_player_fourth_max', 'FG%_rolling_5_player_fourth_max', '3PT%_rolling_5_player_fourth_max', 'FT%_rolling_5_player_fourth_max',
        'ORB_rolling_5_player_fourth_max', 'DRB_rolling_5_player_fourth_max',
        
    ]
    
    # Retrieve final columns used for prediction
    rolling_matchups_inc = matchups[
        ['game_id', 'Date', 'Team_A', 'Team_B', 'is_win_A'] +  # keep winner label
        [f"{c}_A" for c in feature_cols_team_inc] +
        [f"{c}_B" for c in feature_cols_team_inc] +
        [f"{c}_A" for c in feature_cols_players_inc] + 
        [f"{c}_B" for c in feature_cols_players_inc]
    ].copy()
    
    # Relabel the `label` column indicating a win for the home team
    rolling_matchups_inc.rename(columns={"is_win_A": "label"}, inplace=True)
    rolling_matchups_inc['label'] = rolling_matchups_inc['label'].astype(int)
    
    
    rolling_matchups_inc.isnull().sum()
    rolling_matchups_inc.dropna(axis=0, inplace=True)
    
    
    
    X = rolling_matchups_inc.drop(columns=['label', 'game_id', 'Date', 'Team_A', 'Team_B'])
    y = rolling_matchups_inc['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=101)
    x_cols = list(rolling_matchups_inc.drop(columns=['label', 'game_id', 'Date', 'Team_A', 'Team_B']).columns)
    X_train = X_train[x_cols]
    X_test = X_test[x_cols]
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled_inc = scaler.fit_transform(X_train)
    X_test_scaled_inc = scaler.transform(X_test)
    
    # Logistic Regression
    from sklearn.linear_model import LogisticRegression
    
    log_inc = LogisticRegression(random_state=101)
    log_inc.fit(X_train_scaled_inc, y_train)
    pred_y_log_inc = log_inc.predict(X_test_scaled_inc)
    
    
    # KNN
    from sklearn.neighbors import KNeighborsClassifier
    
    knn_inc = KNeighborsClassifier()
    knn_inc.fit(X_train_scaled_inc, y_train)
    pred_y_knn_inc = knn_inc.predict(X_test_scaled_inc)
    
    
    # SVM
    from sklearn.svm import SVC
    
    svc_inc = SVC(random_state=101)
    svc_inc.fit(X_train_scaled_inc, y_train)
    pred_y_svc_inc = svc_inc.predict(X_test_scaled_inc)
    
    
    
    from sklearn.model_selection import GridSearchCV
    
    param_grid_log = {
        'penalty': ['l1', 'l2'],
        'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        'solver': ['liblinear', 'lbfgs'], 
        'max_iter': [100, 200, 500]
    }
    param_grid_knn = {
        'n_neighbors': [3, 5, 7, 8, 9, 11],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan'],
        'leaf_size': [20, 30, 40],
    }
    param_grid_svc = {
        'C': [0.1, 1, 3, 10, 31, 100],
        'kernel': ['linear', 'rbf'],
        'gamma': [0.0001, 0.001, 0.01, 0.1, 1]
    }
    grid_logistic = GridSearchCV(LogisticRegression(random_state=101, n_jobs=-1), param_grid=param_grid_log, refit=True, verbose=0)
    grid_knn = GridSearchCV(KNeighborsClassifier(), param_grid=param_grid_knn, refit=True, verbose=0, scoring='accuracy')
    grid_svc = GridSearchCV(SVC(random_state=101), param_grid=param_grid_svc, refit=True, verbose=0)
    
    grid_logistic.fit(X_train_scaled_inc, y_train)
    grid_knn.fit(X_train_scaled_inc, y_train)
    grid_svc.fit(X_train_scaled_inc, y_train)
    
    
    y_pred_grid_log = grid_logistic.predict(X_test_scaled_inc)
    
    y_pred_grid_knn = grid_knn.predict(X_test_scaled_inc)
    
    y_pred_grid_svc = grid_svc.predict(X_test_scaled_inc)

    return {'grid_logistic': grid_logistic, 'df': df, 
            'feature_cols_team_inc': feature_cols_team_inc, 
            'feature_cols_players_inc': feature_cols_players_inc,
            'scaler': scaler, 
            'mapping': team_mapping}

resources = load_model()
grid_logistic = resources['grid_logistic']
df = resources['df']
feature_cols_team_inc = resources['feature_cols_team_inc']
feature_cols_players_inc = resources['feature_cols_players_inc']
scaler = resources['scaler']
mapping = resources['mapping']


conference_teams = ['goucher',
    'scranton',
    'elizabethtown',
    'catholic',
    'susquehanna',
    'juniata',
    'drew',
    'moravian',
    'lycoming',
    'wilkes']



# Function to predict winner of matchup; returns True for home wins, False for Away wins
def predict(model, home_team_name, away_team_name):
    
    # Get most recent rows
    home_stats = df[df['Team']==home_team_name].sort_values(by='Date').iloc[-1:][feature_cols_team_inc + feature_cols_players_inc]
    away_stats = df[df['Team']==away_team_name].sort_values(by='Date').iloc[-1:][feature_cols_team_inc + feature_cols_players_inc]

    # Split & rename stats
    home_team_stats  = home_stats[feature_cols_team_inc].rename(columns=lambda c: f"{c}_A")
    away_team_stats  = away_stats[feature_cols_team_inc].rename(columns=lambda c: f"{c}_B")

    home_player_stats = home_stats[feature_cols_players_inc].rename(columns=lambda c: f"{c}_A")
    away_player_stats = away_stats[feature_cols_players_inc].rename(columns=lambda c: f"{c}_B")

    # Combine input
    X_in = pd.concat([
        home_team_stats, 
        away_team_stats, 
        home_player_stats, 
        away_player_stats
    ], axis=1)

    # IMPORTANT: fix NaNs before scaling
    X_in = X_in.fillna(0)

    # Scale
    X_scaled = scaler.transform(X_in)

    # Predict
    pred  = model.predict(X_scaled)
    proba = model.predict_proba(X_scaled)

    return pred, proba



# =======================
# STREAMLIT APP
# =======================
# Win Prediction Section
st.title("🏀 Basketball Winner Predictor")
st.write("Select two teams to predict the game outcome based on rolling 5-game stats.")

teamA = st.selectbox("Home Team", conference_teams, key="teamA")
teamB = st.selectbox("Away Team", conference_teams, key="teamB")

# Get most recent rows
home_stats = df[df['Team']==teamA].sort_values(by='Date').iloc[-1:][feature_cols_team_inc + feature_cols_players_inc]
away_stats = df[df['Team']==teamB].sort_values(by='Date').iloc[-1:][feature_cols_team_inc + feature_cols_players_inc]

# Split & rename stats
home_team_stats  = home_stats[feature_cols_team_inc].rename(columns=lambda c: f"{c}_A")
away_team_stats  = away_stats[feature_cols_team_inc].rename(columns=lambda c: f"{c}_B")

 # Combine input
X_in = pd.concat([
    home_team_stats, 
    away_team_stats
], axis=1)

if teamA == teamB:
    st.warning("Teams must be different.")
    st.stop()

if st.button("Predict Winner"):
    pred, proba = predict(grid_logistic, teamA, teamB)

    pred_class = pred[0]
    proba_home = proba[0][1]  # Probability home wins *if classes_ == [0,1]*

    team_winner = teamA if pred_class == 1 else teamB

    st.subheader(f"🏆 Predicted Winner: **{team_winner}**")

# Statistic Graph Section
st.title("📊 Statistical Analysis")
st.write("Show the statistics of the two teams or players.")

with st.expander("Show Graphical Analysis", expanded=False):
    st.subheader("Average Team Stats Comparison")
    
    # --- Compute averages for each team ---
    def team_avg(df, team_name, cols):
        subset = df[df['Team'] == team_name]
        if subset.empty:
            return pd.Series({c: np.nan for c in cols})
        available = [c for c in cols if c in subset.columns]
        avg = subset[available].mean()
        return avg.reindex(cols)

    teamA_stats = team_avg(df, teamA, feature_cols_team_inc)
    teamB_stats = team_avg(df, teamB, feature_cols_team_inc)

    comparison_df = pd.DataFrame({
        'Stat': feature_cols_team_inc,
        teamA: teamA_stats.values,
        teamB: teamB_stats.values
    }).dropna(how='all')

    # --- Readable labels ---
    label_map = {
        'PTS_rolling_5': 'Points',
        'REB_rolling_5': 'Rebounds',
        'A_rolling_5': 'Assists',
        'TO_rolling_5': 'Turnovers',
        'STL_rolling_5': 'Steals',
        'BLK_rolling_5': 'Blocks',
        'PF_rolling_5': 'Fouls',
        'FG%_rolling_5': 'Field Goal %',
        '3PT%_rolling_5': '3PT %',
        'FT%_rolling_5': 'Free Throw %',
        'ORB_rolling_5': 'Off. Rebounds',
        'DRB_rolling_5': 'Def. Rebounds',
        'is_win_rolling_5': 'Win % (last 5)'
    }
    comparison_df['Stat'] = comparison_df['Stat'].map(label_map)
    
    # --- Normalize percentage columns to 0–100 ---
    percent_stats = ['Field Goal %', '3PT %', 'Free Throw %', 'Win % (last 5)']
    for col in [teamA, teamB]:
        comparison_df.loc[comparison_df['Stat'].isin(percent_stats), col] *= 100

    # --- Grouped Bar Chart ---
    fig_bar = px.bar(
        comparison_df.melt(id_vars='Stat', var_name='Team', value_name='Average'),
        x='Stat', y='Average', color='Team', barmode='group',
        title=f"Average Stats: {teamA} vs {teamB}"
    )
    fig_bar.update_layout(xaxis_tickangle=-35)
    st.plotly_chart(fig_bar, use_container_width=True)

# Recent Form Tracker Section
st.title("📈 Recent Form Tracker")
st.write("Track the last 5 games for each team.")

with st.expander("Show Recent Form", expanded=False):
    st.subheader("Average Team Stats Comparison")
    # Helper function to get last 5 games
    def recent_form(df, team_name):
        subset = df[df['Team'] == team_name].sort_values('Date').tail(5)
        # Keep key stats
        return subset[['Date', 'PTS', 'REB', 'A', 'TO', 'STL', 'BLK', 'PF', 'is_win']]

    teamA_form = recent_form(df, teamA)
    teamB_form = recent_form(df, teamB)

    # Display tables
    st.subheader(f"📊 {teamA} - Last 5 Games")
    st.dataframe(teamA_form)

    st.subheader(f"📊 {teamB} - Last 5 Games")
    st.dataframe(teamB_form)

    # Plot win/loss trend
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=teamA_form['Date'], y=teamA_form['PTS'],
        mode='lines+markers', name=f"{teamA} Points",
        line=dict(color='blue')
    ))
    fig_trend.add_trace(go.Scatter(
        x=teamB_form['Date'], y=teamB_form['PTS'],
        mode='lines+markers', name=f"{teamB} Points",
        line=dict(color='red')
    ))
    fig_trend.update_layout(
        title=f"Points Trend (Last 5 Games): {teamA} vs {teamB}",
        xaxis_title="Date",
        yaxis_title="Points",
        legend_title="Team"
    )
    st.plotly_chart(fig_trend, use_container_width=True) 
