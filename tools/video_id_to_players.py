
import pdb
import os
import re
import requests
import pandas as pd
import numpy as np
import json

from bs4 import BeautifulSoup
from random import randint
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

options = Options()
options.add_argument('--headless')
driver = webdriver.Chrome(options=options)
DATA_DIR = "/home/ubuntu/NSVA/tools/pbp_videos/"

game_ids = set()
for split in ["train", "test", "val"]:
    files = os.listdir(DATA_DIR + split)
    for f in files:
        if not f.endswith("mp4"): continue
        game_id = f.split("/")[-1].split("-")[0]
        game_ids.add(game_id)
print(len(list(game_ids)), "Games found")

player_id_dict = {}
df = pd.read_csv("NBA_Player_IDs.csv", encoding='unicode_escape')
for i in range(len(df)):
    row = df.iloc[i]
    if np.isnan(row["NBAID"]): continue
    player_id_dict[row["NBAName"]] = int(row["NBAID"])

game_to_players_dict = {}
for game_id in list(game_ids):
    player_ids = []
    url = "https://www.nba.com/game/" + str(game_id) + "/box-score"
    print(url)
    driver.get(url)
    res = driver.page_source
    soup = BeautifulSoup(res, 'html.parser')
    out = soup.findChildren('span', {"class":"GameBoxscoreTablePlayer_gbpNameFull__cf_sn"})
    for x in out:
        player_ids.append(player_id_dict[x.contents[0]])
    game_to_players_dict[game_id] = player_ids

with open('data.json', 'w') as fp:
    json.dump(game_to_players_dict, fp)

