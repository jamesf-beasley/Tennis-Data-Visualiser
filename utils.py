import os
import json

def save_data_to_file(player_url, data):
    """Save the player data to a JSON file.

    Args:
        player_url: The URL of the player for the file.
        data (dict): The data to be saved.
    """
    directory = "scraped_data"
    os.makedirs(directory, exist_ok=True)
    player_url = player_url.split('=')[-1]
    filename = os.path.join(directory, f"{player_url}.json")

    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)
    print(f"Data saved to {filename}")

def load_data_from_file(player_url):
    """Load player data from a JSON file.

    Args:
        player_url: The URL of the player to find the file.

    Returns:
        data or None: The loaded data if found, or None if no data exists.
    """
    directory = "scraped_data"
    player_url = player_url.split('=')[-1]
    filename = os.path.join(directory, f"{player_url}.json")

    if os.path.exists(filename):
        with open(filename, 'r') as file:
            data = json.load(file)
        print(f"Data loaded from {filename}")
        return data
    else:
        print(f"No data found for {player_url} at {filename}")
        return None