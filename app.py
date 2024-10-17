from flask import Flask, render_template, request, redirect, url_for
from web_scrapereal import get_player_data, get_initial_players
from plotting import plot_data_h2h, plot_data
import pandas as pd
from utils import load_data_from_file
from model import predict_winner

app = Flask(__name__)

@app.route('/')
def landing():
    """ Initially renders the landing page."""
    return render_template('landing.html')

@app.route('/category', methods=['POST'])
def select_category():
    """Redirect to the appropriate page based on the selected category.

    Returns:
        redirect: the url to redirect to the corresponding page.
        str: An error message.
    """
    category = request.form.get('category')
    print(f"Category is sc: {category}")
    if category == 'WTA':
        return redirect(url_for('index', category='WTA'))
    elif category == 'LTA':
        return redirect(url_for('index', category='LTA'))
    else:
        return "Invalid category selected.", 400

@app.route('/index')
@app.route('/index/<category>')
def index(category):
    """Renders the index page with a list of players for the selected category (e.g. WTA).

    Args:
        category (str): The category of players (either WTA or LTA).

    Returns:
        render_template: The rendered index page with players of the select category.
    """
    players = list(get_initial_players(category).keys())
    return render_template('index.html', players=players, category=category)

@app.route('/get_player_data', methods=['POST'])
def get_player_data_route():
    """Retrieve and plot player data based on the selected player.

    Returns:
        render_template: All of the corresponding player data 
        str: An error message.
    """
    player_name = request.form.get('player')
    plot_type = request.form.get('plot_type', 'radar')
    category = request.form.get('category')
    players = list(get_initial_players(category).keys())
    player_url = get_initial_players(category).get(player_name)
    if player_url:
        player_url_toname = player_url.split('=')[-1]
        data_file = load_data_from_file(player_url_toname)
        if data_file is not None:
            data = data_file
        else:
            data = get_player_data(player_url)
        plot_url = plot_data(data, category, plot_type)  
        total_wins, total_losses = get_player_stats(data)
        return render_template('player_data.html', player_name=player_name, plot_url=plot_url, players=players, total_wins=total_wins, total_losses=total_losses, category=category)
    else:
        return "Player data not found.", 404

def get_player_stats(data):
    """Calculates a player's win to loss ratio.

    Args:
        data (dict): The corresponding player data.

    Returns:
        win_num, loss_num: Total number of wins and losses.
    """
    if 'career-splits' in data:
        df = pd.DataFrame(data['career-splits']['data'], columns=data['career-splits']['headers'])
    else:
        df = pd.DataFrame(data['career-splits-chall']['data'], columns=data['career-splits-chall']['headers'])
    df = df.head(4)
    headers = ['M','L']
    df = df[headers].fillna(0)
    df['M'] = pd.to_numeric(df['M'], errors='coerce')
    win_num = df['M'].sum()
    df['L'] = pd.to_numeric(df['L'], errors='coerce')
    loss_num = df['L'].sum()
    return win_num, loss_num

@app.route('/head_to_head', methods=['POST', 'GET'])
def head_to_head():
    """Render the head-to-head page if selected.

    Returns:
        render_template: All of the data for both players.
        str: An error message.
    """
    category = request.form.get('category')
    print(f"Category is h2h: {category}")
    players = list(get_initial_players(category).keys())

    if request.method == 'POST':
        player1_name = request.form.get('player1')
        player2_name = request.form.get('player2')
        player1_url = get_initial_players(category).get(player1_name)
        player2_url = get_initial_players(category).get(player2_name)
        plot_type = request.form.get('plot_type_h2h', 'radar_h2h')

        if player1_url and player2_url:
            player1_url_toname = player1_url.split('=')[-1]
            player2_url_toname = player2_url.split('=')[-1]
            player1_datafile = load_data_from_file(player1_url_toname)
            player2_datafile = load_data_from_file(player2_url_toname)

            if player1_datafile is not None:
                player1_data = player1_datafile
            else:
                player1_data = get_player_data(player1_url)
            if player2_datafile is not None:
                player2_data = player2_datafile
            else:
                player2_data = get_player_data(player2_url)

            wins, losses = head_to_head_results(player1_data, player2_data, player1_name, player2_name)
            predicted_winner = predict_winner(player1_data, player2_data, player1_name, player2_name)
            plot_url = plot_data_h2h(player1_data, player2_data, player1_name, player2_name, category, plot_type)
            return render_template('head_to_head.html', player1_name=player1_name, player2_name=player2_name, plot_url=plot_url, players=players, wins=wins, losses=losses, predicted_winner=predicted_winner, category=category)
        else:
            return "Player data not found.", 404
    else:
        return render_template('head_to_head.html', players=players, category=category)

def head_to_head_results(player1_data, player2_data, player1_name, player2_name):
    """Calculate the head-to-head win to loss ratio between the players.

    Args:
        player1_data: The data for player 1.
        player2_data: The data for player 2.
        player1_name: The name of player 1.
        player2_name: The name of player 2.

    Returns:
        wins and losses: Number of wins and losses for player 1 against player 2.
    """
    df1 = pd.DataFrame(player1_data['head-to-heads']['data'], columns=player1_data['head-to-heads']['headers']).iloc[:, 1:4]
    df2 = pd.DataFrame(player2_data['head-to-heads']['data'], columns=player2_data['head-to-heads']['headers']).iloc[:, 1:4]
    df1['Opponent'] = df1['Opponent'].str.split('[').str[0].str.strip()
    df2['Opponent'] = df2['Opponent'].str.split('[').str[0].str.strip()
    df1_filtered = df1[df1['Opponent'] == player2_name]
    df2_filtered = df2[df2['Opponent'] == player1_name]
    if not df1_filtered.empty and not df2_filtered.empty:
        wins = df1_filtered['W'].values[0]
        losses = df1_filtered['L'].values[0]
        return wins, losses
    elif not df1_filtered.empty and df2_filtered.empty:
        wins = df1_filtered['W'].values[0]
        losses = df1_filtered['L'].values[0]
        return wins, losses
    elif df1_filtered.empty and not df2_filtered.empty:
        wins = df2_filtered['W'].values[0]
        losses = df2_filtered['L'].values[0]
        return losses, wins
    else:
        return None, None

if __name__ == '__main__':
    app.run(debug=True)