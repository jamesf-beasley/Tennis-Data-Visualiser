import requests
from bs4 import BeautifulSoup
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import matplotlib
matplotlib.use('Agg')
from utils import save_data_to_file


def get_initial_players(category='WTA'):
    """Scrape player data based on the category (WTA or ATP).

    Args:
        category: The chosen category of the players.

    Returns:
        player_links: A dictionary of player names and their respective URLs.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/104.0.0.0 Safari/537.36',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
        'Connection': 'keep-alive',
    }

    if category == 'WTA':
        url = 'https://tennisabstract.com/reports/wta_elo_ratings.html'
        print('scrape', url)
    else:
        url = 'https://tennisabstract.com/reports/atp_elo_ratings.html'
        print('scrape', url)
    
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table', {'id': 'reportable'})

    player_links = {}

    for row in table.find_all('tr'):
        cells = row.find_all('td')
        for cell in cells:
            link = cell.find('a')
            if link:
                player_name = link.text.strip().replace('\xa0', ' ')
                player_link = link.get('href')
                player_links[player_name] = player_link

    return player_links


def get_player_data(player_url):
    """Scrape player data from their page.

    Args:
        player_url: The URL of the player's page.

    Returns:
        all_data: A dictionary of data about the player.
    """
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)

    driver.get(player_url)
    time.sleep(5)

    table_ids = ['year-end-rankings', 'head-to-heads', 'recent-results']
    all_data = {}

    for table_id in table_ids:
        try:
            table = driver.find_element(By.ID, table_id)
            headers = [header.text for header in table.find_elements(By.TAG_NAME, 'th')]
            rows = table.find_elements(By.TAG_NAME, 'tr')
            table_data = []

            for row in rows:
                cells = row.find_elements(By.TAG_NAME, 'td')
                row_data = [cell.text.strip() for cell in cells]
                if row_data:
                    table_data.append(row_data)

            if table_data:
                all_data[table_id] = {'headers': headers, 'data': table_data}
        except Exception:
            print(f"An error occurred while scraping table with ID '{table_id}'")
    
    table_ids2 = ['career-splits', 'last52-splits', 'tour-years']
    table_id_found = False
    
    for table_id in table_ids2:
        try:
            table = driver.find_element(By.ID, table_id)
            headers = [header.text for header in table.find_elements(By.TAG_NAME, 'th')]
            rows = table.find_elements(By.TAG_NAME, 'tr')
            table_data = []

            for row in rows:
                cells = row.find_elements(By.TAG_NAME, 'td')
                row_data = [cell.text.strip() for cell in cells]
                if row_data:
                    table_data.append(row_data)

            if table_data:
                all_data[table_id] = {'headers': headers, 'data': table_data}
                table_id_found = True
        except Exception:
            print(f"Table with ID '{table_id}' not found. Trying next option.")

    if not table_id_found:
        table_ids_fallback = ['career-splits-chall', 'last52-splits-chall', 'chall-years']
        
        for table_id in table_ids_fallback:
            try:
                table = driver.find_element(By.ID, table_id)
                headers = [header.text for header in table.find_elements(By.TAG_NAME, 'th')]
                rows = table.find_elements(By.TAG_NAME, 'tr')
                table_data = []

                for row in rows:
                    cells = row.find_elements(By.TAG_NAME, 'td')
                    row_data = [cell.text.strip() for cell in cells]
                    if row_data:
                        table_data.append(row_data)

                if table_data:
                    all_data[table_id] = {'headers': headers, 'data': table_data}

            except Exception:
                print(f"An error occurred while scraping fallback table with ID '{table_id}'.")

    try:
        h1_element = driver.find_element(By.ID, 'recent-results-h')
        a_tag = h1_element.find_element(By.TAG_NAME, 'a')
        href = a_tag.get_attribute('href')

        driver.get(href)
        time.sleep(5)

        try:
            matches_table = driver.find_element(By.ID, 'matches')
            headers = [header.text for header in matches_table.find_elements(By.TAG_NAME, 'th')]
            rows = matches_table.find_elements(By.TAG_NAME, 'tr')
            matches_data = []

            for row in rows:
                cells = row.find_elements(By.TAG_NAME, 'td')
                row_data = [cell.text.strip() for cell in cells]
                if row_data:
                    matches_data.append(row_data)

            if matches_data:
                all_data['matches'] = {'headers': headers, 'data': matches_data}
            else:
                print("No data found in the 'matches' table.")

        except Exception:
            print(f"An error occurred while scraping the 'matches' table.")

    except Exception:
        print(f"An error occurred while finding the href in h1 with ID 'recent-results-h'.")

    driver.quit()
    save_data_to_file(player_url, all_data)
    return all_data


