import asyncio
from playwright.sync_api import sync_playwright
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
import pandas as pd
from functools import reduce
from time import sleep
import random
import os

def scrape_game(url):
    # --- Step 1: Fetch fully-rendered HTML with Playwright ---
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)  # headless browser
        page = browser.new_page()
        page.goto(url, wait_until="networkidle", timeout=120000)    # wait until JS loads tables
        html = page.content()
        browser.close()

    soup = BeautifulSoup(html, "html.parser")
    print('URL loaded successfully.')


    # --- Step 2: Extract game metadata ---
    game_details_element = soup.find("aside", class_="game-details")
    game_details = {}
    key = None
    for detail in game_details_element.dl.children:
        if detail.name is not None:
            if detail.name != 'dt':
                game_details[key] = detail.get_text(strip=True)
            else:
                key = detail.get_text(strip=True)
    game_details['Date'] = game_details['Date'].replace('/', '-')
    game_date = game_details['Date']
    
    # Get total scores and names for teams (totals are first two of six headers)
    team_names_and_scores = [h3.get_text(strip=True) for h3 in soup.find_all("h3", class_="sub-heading")][:2]
    if len(team_names_and_scores) != 2:
        raise ValueError("Could not find both team names")

    home_team, home_score = " ".join(team_names_and_scores[0].split(" ")[:-1]), team_names_and_scores[0].split(" ")[-1]
    away_team, away_score = " ".join(team_names_and_scores[1].split(" ")[:-1]), team_names_and_scores[1].split(" ")[-1]
    game_id = f"{home_team}_{away_team}_{game_date.replace(' ', '_')}"
    print(f"Game teams: {home_team} ({home_score}) vs. {away_team} ({away_score})")
    
    # --- Step 3: Extract player stats ---
    player_columns = ["##", "Player", "GS", "MIN", "FG", "3PT", "FT",
                      "ORB-DRB", "REB", "PF", "A", "TO", "BLK", "STL", "PTS"]

    all_players = []

    for idx, team in enumerate([home_team, away_team]):
        table_id = f"DataTables_Table_{idx}"
        table = soup.find("table", id=table_id)
        if not table:
            continue

        rows = table.find("tbody").find_all("tr")
        for row in rows:
            cols = [c.get_text(strip=True) for c in row.find_all(['td', 'th'])]
            if not cols or "TMTEAM" in cols[1]:
                continue

            player_data = dict(zip(player_columns, cols))
            # Remove jersey number from name
            player_data['Player'] = "".join([char for char in player_data['Player'] if not char.isdigit()])
            player_data["Team"] = team
            player_data["game_id"] = game_id
            all_players.append(player_data)

    player_df = pd.DataFrame(all_players)
    player_df.to_csv(f"player_game_stats/player_data_{game_id}.csv", index=False)

    # --- Step 4: Extract team totals ---
    totals_data = []
    for idx, team in enumerate([home_team, away_team]):
        table_id = f"DataTables_Table_{idx}"
        table = soup.find("table", id=table_id)
        rows = table.find("tfoot").find_all("tr")
        

        for row in rows:
            cols = [c.get_text(strip=True) for c in row.find_all(["td","th"])]
            if cols and cols[1] == "Totals":
                totals_dict = {
                    "game_id": game_id,
                    "Team": team,
                    "Home/Away": "home" if idx == 0 else "away",
                }
                totals_dict.update(game_details)
                for key, value in zip(player_columns[3:], cols[3:]):  # skip ##, Player, GS
                    totals_dict[key] = value
                totals_data.append(totals_dict)

    totals_df = pd.DataFrame(totals_data)
    totals_df.to_csv(f"team_game_stats/game_data_{game_id}.csv", index=False)

    print(f"Saved: player_game_stats/player_data_{game_id}.csv and team_game_stats/game_data_{game_id}.csv")

def scrape_individual_game_urls(url):
    """
    Fetches a webpage and extracts all individual game URLs from the results table.

    The function:
    - Loads the page using Playwright (so JavaScript is executed)
    - Finds the section with id="res-overal"
    - Grabs the first <table> within that section
    - Iterates through each <tr> in the <tbody>
    - Extracts the first <td> text (or hyperlink if present)
    - Returns a list of game URL strings
    """
    game_urls = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, wait_until="networkidle")
        html = page.content()
        browser.close()

    soup = BeautifulSoup(html, "html.parser")

    section = soup.find("section", id="res-overall")
    if not section:
        print("Section with id='res-overall' not found.")
        return []

    table = section.find("table")
    if not table:
        print("No table found inside the section.")
        return []

    tbody = table.find("tbody")
    if not tbody:
        print("No tbody found inside the table.")
        return []

    rows = tbody.find_all("tr")
    for row in rows:
        first_td = row.find("td")
        if not first_td:
            continue

        # If there's a link in the first cell, extract href
        link_tag = first_td.find("a")
        if link_tag and link_tag.has_attr("href"):
            href = link_tag["href"]
            # Ensure full URL (handle relative links)
            if href.startswith("http"):
                game_urls.append(href)
            else:
                game_urls.append("https://landmarkconference.org/" + href)
        else:
            # fallback to text content
            game_urls.append(first_td.get_text(strip=True))

    print(f"Found {len(game_urls)} game URLs.")
    return game_urls


async def scrape_team_season(url, school, year):
    """
    Scrapes a team's season statistics table from a given URL and saves it as a CSV.
    
    Parameters:
        url (str): The page URL containing the team's seasonal stats.
        school (str): The school's name (used for filename and team column).
        year (str or int): The season year (used for filename).
    """
    print(f"\nFetching season stats for {school} ({year}) ...")

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False)
            context = await browser.new_context(
                user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
            )
            page = await context.new_page()
            await page.goto(url, wait_until="networkidle", timeout=30000)
            print("stalling")
            await page.wait_for_timeout(150000)  # wait extra 5 seconds for JS to load
            html = await page.content()
            await browser.close()
    except Exception as e:
        print(f"⚠️  Could not load page for {school} {year}: {e}")
        return None

    soup = BeautifulSoup(html, "html.parser")

    # Find the single main stats table
    table = soup.find("table")
    if not table:
        print(f"⚠️  No stats table found for {school} ({year}). Skipping.")
        return None

    tbody = table.find("tbody")
    if not tbody:
        print(f"⚠️  Table has no <tbody> for {school} ({year}). Skipping.")
        return None

    rows = tbody.find_all("tr")
    if not rows:
        print(f"⚠️  No data rows found for {school} ({year}). Skipping.")
        return None

    # Define expected columns
    columns = [
        "School", "Name", "GP", "MP", "PTS", "PPG", "FGM", "FGA", "FG%", 
        "3PM", "3PA", "3P%", "FTM", "FTA", "FT%", "OREB", "DREB", 
        "REB/G", "AST", "AST/G", "STL", "BLK"
    ]

    data = []

    for row in rows:
        th = row.find("th")
        tds = row.find_all("td")

        if not th or not tds:
            continue

        player_name = th.get_text(strip=True)
        values = [td.get_text(strip=True) for td in tds]

        if len(values) != len(columns) - 2:  # sanity check for column count
            print(f"Skipping malformed row for player: {player_name}")
            continue

        # Prepend school and player name
        row_data = [school, player_name] + values
        data.append(row_data)

    # Build DataFrame
    df = pd.DataFrame(data, columns=columns)

    # Ensure directory exists
    os.makedirs("season_stats", exist_ok=True)

    # Sanitize filename
    safe_school = "".join(c for c in school if c.isalnum() or c in ('_', '-')).replace(" ", "_")
    csv_path = f"season_stats/{safe_school}_{year}_seasonal_stats.csv"

    df.to_csv(csv_path, index=False)
    print(f"✅ Saved {len(df)} players to {csv_path}")

    return df

import requests
import os
import pandas as pd
from urllib.parse import urlparse, parse_qs

import os
import re
import requests
import pandas as pd
from bs4 import BeautifulSoup

def scrape_team_season2(url, school, year):
    """
    Scrapes a team's season statistics from a Landmark Conference team page.
    Automatically parses embedded JavaScript to extract API parameters.
    Saves results into season_stats/{school}_{year}_seasonal_stats.csv
    """

    print(f"\nFetching season stats for {school} ({year}) ...")

    # Try to load the page
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code != 200:
            print(f"⚠️ Page request failed ({resp.status_code}) for {school} {year}")
            return None
    except Exception as e:
        print(f"⚠️ Could not fetch page for {school} {year}: {e}")
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    # Look for script element containing conf_stats.ashx
    script_tag = soup.find("script", string=re.compile(r"conf_stats\.ashx"))
    if not script_tag or not script_tag.string:
        print(f"⚠️ Could not find script with conf_stats.ashx for {school} {year}")
        return None

    script_text = script_tag.string

    # Extract parameters from JS
    params = {}
    for key in ["team_id", "sport", "year", "conf", "postseason"]:
        match = re.search(rf"{key}:\s*'([^']+)'", script_text)
        if match:
            params[key] = match.group(1)

    if not params.get("team_id"):
        print(f"⚠️ Could not extract team_id for {school} {year}")
        return None

    # Construct API endpoint
    api_url = (
        "https://landmarkconference.org/services/conf_stats.ashx"
        f"?method=get_team_stats"
        f"&team_id={params['team_id']}"
        f"&sport={params.get('sport', 'wbball')}"
        f"&year={params.get('year', year)}"
        f"&conf={params.get('conf', 'False')}"
        f"&postseason={params.get('postseason', 'False')}"
    )

    # Request data from API
    try:
        res = requests.get(api_url, timeout=30)
        res.raise_for_status()
        data_json = res.json()
    except Exception as e:
        print(f"⚠️ Failed to fetch JSON stats for {school} {year}: {e}")
        return None

    # Verify data
    players = data_json.get("players") or data_json.get("aaData")
    if not players:
        print(f"⚠️ No player data returned for {school} {year}")
        return None

    # Define columns
    columns = [
        "School", "Name", "GP", "MP", "PTS", "PPG", "FGM", "FGA", "FG%", 
        "3PM", "3PA", "3P%", "FTM", "FTA", "FT%", "OREB", "DREB", 
        "REB/G", "AST", "AST/G", "STL", "BLK"
    ]

    # Parse player data
    data_rows = []
    for player in players:
        # Each player has stats in nested dicts like player["stats_stats"]
        stats = player.get("stats_stats", {})
        name = player.get("name", "").strip()
        row = [
            school,
            name,
            stats.get("games_played", ""),
            stats.get("minutes_played", ""),
            stats.get("points", ""),
            stats.get("points_per_game", ""),
            stats.get("field_goals_made", ""),
            stats.get("field_goals_attempted", ""),
            stats.get("field_goal_percentage", ""),
            stats.get("three_point_made", ""),
            stats.get("three_point_attempted", ""),
            stats.get("three_point_percentage", ""),
            stats.get("free_throws_made", ""),
            stats.get("free_throws_attempted", ""),
            stats.get("free_throw_percentage", ""),
            stats.get("offensive_rebounds", ""),
            stats.get("defensive_rebounds", ""),
            stats.get("rebounds_per_game", ""),
            stats.get("assists", ""),
            stats.get("assists_per_game", ""),
            stats.get("steals", ""),
            stats.get("blocks", "")
        ]
        data_rows.append(row)

    # Save to CSV
    os.makedirs("season_stats", exist_ok=True)
    safe_school = "".join(c for c in school if c.isalnum() or c in ('_', '-')).replace(" ", "_")
    csv_path = f"season_stats/{safe_school}_{year}_seasonal_stats.csv"
    df = pd.DataFrame(data_rows, columns=columns)
    df.to_csv(csv_path, index=False)

    print(f"✅ Saved {len(df)} player stats to {csv_path}")
    return df


# Example usage
if __name__ == "__main__":
    
    # Example: scrape all game URLs from the team results page
    # year = 2020
    # for year in range(2020, 2025):
    for year in range(2022, 2025):
    
        schedule_url = f"https://landmarkconference.org/stats.aspx?path=wbball&year={year}"
        game_links = scrape_individual_game_urls(schedule_url)
        # print(game_links)

        # # Optionally scrape each game found
        for link in game_links:
            scrape_game(link)
            sleep(0.15+random.randint(1, 20))
            
            print("Downloaded game.")
        print(f"Completed scraping for year {year}.")




    # FAILED ATTEMPT AT GETTING SEASON DATA

    # school = "Scranton"
    # year = "2024"
    # season_url = f"https://landmarkconference.org/teamstats.aspx?path=wbball&year={year}&school={school}"

    # asyncio.run(scrape_team_season(season_url, school, year))


    exit()

    # # Test game urls
    # #url = "https://landmarkconference.org/boxscore.aspx?id=7ltyipP022aUJKmSgYVs3wXi7up73ghniwD5ElWyd07J7BNseBaMQePf0fY%2f3yUPeexHLry75Bk8B4haDUktrKmPOC9XErB9%2b1dXtF%2fnymgbqY%2beyP3Fwtad%2boY72j0EbH3Pbwom8%2f4cf%2fSLuMMrPXyTyKwD4PF4M2%2fALfPALok%3d&path=wbball"
    # url = "https://landmarkconference.org/boxscore.aspx?id=7ltyipP022aUJKmSgYVs3wXi7up73ghniwD5ElWyd07J7BNseBaMQePf0fY%2f3yUPhQgPJ%2b7GCEjYMd0kjOdG3LvSfCWANMKZBlFuMUogrBBxBkPh%2fwm%2bT0ijMPqPkH%2bN3ynL%2fRKiuM0ZF7wfbaRogiq77o9zxqA4p9Ic%2bxkPhC4%3d&path=wbball"
    # scrape_game(url)
