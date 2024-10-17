import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

def get_players_data(player1_data, player2_data, player1_name, player2_name):
    """Extract and process the players' data.

    Args:
        player1_data: The data for player 1.
        player2_data: The data for player 2.
        player1_name: The name of player 1.
        player2_name: The name of player 2.

    Returns:
        df1, df2: Match results for player 1 and player 2.
        None, None: Nothing if data not found.
    """
    if 'matches' in player1_data and 'matches' in player2_data:
        df1 = pd.DataFrame(player1_data['matches']['data'], columns=player1_data['matches']['headers'])
        df1.columns = [col.replace('\n', ' ') for col in df1.columns]
        df2 = pd.DataFrame(player2_data['matches']['data'], columns=player2_data['matches']['headers'])
        df2.columns = [col.replace('\n', ' ') for col in df2.columns]
        
        headings = ['Rk', 'vRk', ""]
        headings2 = ['Rk', 'vRk', "Results"]
        df1 = df1[headings]
        df2 = df2[headings]
        results_column = ""
        player1_name = player1_name.split(' ', 1)[1].lower().strip()
        player2_name = player2_name.split(' ', 1)[1].lower().strip()
        
        print(f"Player 1 Name: {player1_name}")
        print(f"Player 2 Name: {player2_name}")

        def determine_result(row, player_name):
            """Determine the match result for a given player.

            Args:
                row (pd.Series): A row of the dataFrame.
                player_name (str): The name of the selected player.

            Returns:
                W/L/"": 'W' if the player won, 'L' if lost, or empty string if no result.
            """
            result_str = row[results_column].lower().strip()
            cheese = str(result_str.split('[')[:1]).strip()

            if player_name in cheese:
                return 'W'
            elif result_str == "":
                return ""
            else:
                return 'L'

        df1['Results'] = df1.apply(determine_result, axis=1, player_name=player1_name)
        df1 = df1[headings2]
        df1['Player'] = player1_name

        df2['Results'] = df2.apply(determine_result, axis=1, player_name=player2_name)
        df2 = df2[headings2]
        df2['Player'] = player2_name

        return df1, df2
    else:
        return None, None

def prepare_data(df1, df2): 
    """Prepare and preprocess data.

    Args:
        df1: Player 1's dataframe.
        df2: Player 2's dataframe.

    Returns:
        df: A combined dataframe.
    """
    df = pd.concat([df1, df2], ignore_index=True)
    
    print("Data before encoding and mapping:")
    print(df.head())
    print(df['Player'].value_counts())
    print(df['Results'].value_counts())
    
    label_encoder = LabelEncoder()
    df['Player'] = label_encoder.fit_transform(df['Player'])
    df['Results'] = df['Results'].map({'W': '1', 'L': '0'})
    
    df.dropna(subset=['Results'], inplace=True)
    
    print("Data after encoding and mapping:")
    print(df.dtypes)
    print(df.isnull().sum())
    
    return df

def train_model(df):
    """Train a Random Forest model.
    Args:
        df: Combined player dataframe.

    Returns:
        model: The trained model.
    """
    X = df.drop('Results', axis=1)
    y = df['Results']
    
    X = X.apply(pd.to_numeric, errors='coerce')
    y = y.apply(pd.to_numeric, errors='coerce')
    
    X.dropna(inplace=True)
    y = y[X.index]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(random_state=42)
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"Accuracy: {accuracy}")
    print(f"Classification Report:\n{report}")

    cv_scores = cross_val_score(model, X, y, cv=5)
    print(f"Cross-Validation Scores: {cv_scores}")
    print(f"Mean CV Score: {cv_scores.mean()}")
    
    return model

def predict_win(model, df, player1_name, player2_name):
    """Predicts the winner between two players.

    Args:
        model: The trained model.
        df: Combined player dataframe.
        player1_name: The name of player 1.
        player2_name: The name of player 2.

    Returns:
        winner: The name of the predicted winner.
    """
    X = df.drop(columns=['Results'])
    X = X.apply(pd.to_numeric, errors='coerce')
    X.dropna(inplace=True)

    df = df.loc[X.index]

    probabilities = model.predict_proba(X)

    print(f"DataFrame used for prediction:\n{df.head()}")
    print(f"Feature matrix (X):\n{X.head()}")
    print(f"Probabilities shape: {probabilities.shape}")
    print(f"Probabilities:\n{probabilities}")

    player1_indices = df['Player'] == 0
    player2_indices = df['Player'] == 1

    if not player1_indices.any() or not player2_indices.any():
        raise ValueError("Player indices are invalid. Ensure both players are represented in the data.")

    avg_prob_player1 = probabilities[player1_indices][:, 1].mean()
    avg_prob_player2 = probabilities[player2_indices][:, 1].mean()
    
    print(f"Player 1 Index Count: {player1_indices.sum()}")
    print(f"Player 2 Index Count: {player2_indices.sum()}")
    print(f"Average probability of Player 1 winning: {avg_prob_player1}")
    print(f"Average probability of Player 2 winning: {avg_prob_player2}")

    if avg_prob_player1 > avg_prob_player2:
        winner = player1_name
    else:
        winner = player2_name
    
    print(f"Predicted Winner: {winner}")
    return winner

def predict_winner(player1_data, player2_data, player1_name, player2_name):
    """Predict the winner.

    Args:
        player1_data: The data for player 1
        player2_data: The data for player 2.
        player1_name: The name of player 1.
        player2_name: The name of player 2.

    Returns:
        winner: The name of the predicted winner.
    """
    df1, df2 = get_players_data(player1_data, player2_data, player1_name, player2_name)
    df = prepare_data(df1, df2)
    model = train_model(df)
    winner = predict_win(model, df, player1_name, player2_name)
    print(f"Predicted Winner: {winner}")
    return winner