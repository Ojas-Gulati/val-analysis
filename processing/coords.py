import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

with open('./fulldata.json') as f:
    data = json.load(f)

maps_dict = {}
agents_dict = {}

split_data = []
teams = ["red", "blue"]
for match in data:
    for team in teams:
        datapoint = match[team]
        datapoint["attacking"] = (team == "red")
        if datapoint["avg_rank"] is not None and not (np.isnan(datapoint["avg_rank"])):
            # we can process this
            split_data.append(datapoint)
            maps_dict[datapoint["map"]] = True
            for agent in datapoint["agents"]:
                agents_dict[agent] = True

agents = []
maps = []
for agent in agents_dict:
    agents.append(agent)
for map in maps_dict:
    maps.append(map)
maps = sorted(maps)
agents = sorted(agents)

for x in range(len(agents)):
    agents_dict[agents[x]] = x
for x in range(len(maps)):
    maps_dict[maps[x]] = x

print(str(len(split_data)) + " records collected")
print(agents_dict)
print(maps_dict)

datasize = len(split_data)
agentssize = len(agents)
mapssize = len(maps)

x_agents = np.zeros((datasize, agentssize))
x_maps = np.zeros((datasize, mapssize))
x_rank = np.zeros((datasize, ))
x_attacking = np.zeros((datasize, ))

y_winning = np.zeros((datasize, ))

ranks = []
for x in range(datasize):
    data = split_data[x]
    curr_agents = np.zeros((agentssize,))
    for agent in data["agents"]:
        curr_agents[agents_dict[agent]] = 1
    x_agents[x] = curr_agents

    curr_maps = np.zeros((mapssize,))
    curr_maps[maps_dict[data["map"]]] = 1
    x_maps[x] = curr_maps
    x_rank[x] = data["avg_rank"]
    x_attacking[x] = data["attacking"]
    y_winning[x] = data["win"]

np.savez("./processed.npz", x_agents=x_agents, x_maps=x_maps, x_rank=x_rank, x_attacking=x_attacking, y_winning=y_winning)

# ../data/slim_data/x.json

slim_round_win_data = []

from pathlib import Path

pathlist = Path("../data/slim_data").glob('*.json')
total_matches = 0
for path in pathlist:
    # because path is object not string
    pathstr = str(path)
    # read in the data and add the rounds data
    with open(pathstr, 'r', encoding="utf8") as myfile:
        data = json.load(myfile)
        total_matches += len(data)
        for match in data:
            if (match["teams"]["red"] is not None) and (match["teams"]["blue"] is not None) and len(match["rounds"]) != 0:
                red_won = match["teams"]["red"]["has_won"]
                blue_won = match["teams"]["blue"]["has_won"]

                rounds = match["rounds"]
                round_arr = np.zeros((len(rounds),))
                for idx, round_ in enumerate(rounds):
                    winner = round_["winning_team"].lower() == "red"
                    winner = 1 if winner else 0
                    round_arr[idx] = winner
                slim_round_win_data.append((red_won, blue_won, round_arr))

print(len(slim_round_win_data))
print(total_matches)
print(slim_round_win_data[0])

# ../data/slim_data/x.json

round_plant_data = []

from pathlib import Path

rankslist = []
pathlist = Path("../data/slim_data").glob('*.json')
for path in pathlist:
    # because path is object not string
    pathstr = str(path)
    # read in the data and add the rounds data
    with open(pathstr, 'r', encoding="utf8") as myfile:
        data = json.load(myfile)
        for match in data:
            rank_tot = 0
            ranked_players = 0
            for player in match["players"]["all_players"]:
                if (player["currenttier"] != 0):
                    rank_tot += player["currenttier"]
                    ranked_players += 1
            if ranked_players != 0:
                match_rank = rank_tot / ranked_players
                rankslist.append(match_rank)
                for round_ in match["rounds"]:
                    if round_["bomb_planted"]:
                        round_["map"] = match["metadata"]["map"]
                        round_["rank"] = match_rank
                        round_plant_data.append(round_)

rankslist = np.array(rankslist)

print(len(round_plant_data))
print(round_plant_data[0])
print(np.min(rankslist))
print(np.max(rankslist))
print(np.median(rankslist))

# Find out which plant sites are the favourites
# we can use our map lookup from before?

plant_rates = np.zeros((len(maps), 3)) # if they add a map with 4 sites i probably no longer care about the game enough to do this

# loop through our rounds
for _round in round_plant_data:
    map_idx = maps_dict[_round["map"]]
    site_idx = ord(_round["plant_events"]["plant_side"].lower()) - ord("a")
    plant_rates[map_idx][site_idx] += 1

# split plant locations into x_y
plant_counts = np.zeros((len(maps), 3))
plant_locations = []
for x in range(len(maps)):
    sites = []
    for y in range(3):
        sites.append(np.zeros((2, int(np.sum(plant_rates[x][y])))))
    plant_locations.append(sites)

for _round in round_plant_data:
    map_idx = maps_dict[_round["map"]]
    site_idx = ord(_round["plant_events"]["plant_side"].lower()) - ord("a")
    arr_idx = int(plant_counts[map_idx][site_idx])
    plant_locations[map_idx][site_idx][0][arr_idx] = _round["plant_events"]["plant_location"]["x"]
    plant_locations[map_idx][site_idx][1][arr_idx] = _round["plant_events"]["plant_location"]["y"]
    plant_counts[map_idx][site_idx] += 1

# there are 6 ranks to look at: iron, bronze, silver, gold, plat, diamond-immo-radiant
rank_groups = [
    (3, 6, "iron"),
    (6, 9, "bronze"),
    (9, 12, "silver"),
    (12, 15, "gold"),
    (15, 18, "platinum"),
    (18, 30, "diamond+"),
    (30, 45, "all")
]
rank_groups.reverse()

def findRankGroup(rank):
    rank_idx = None
    for idx, (minR, maxR, name) in enumerate(rank_groups):
        if minR <= rank and rank < maxR:
            return idx
    raise Exception("Rank out of bounds")

pixel_size = 5
plant_histograms = np.zeros((len(maps), 3), dtype=object)

for x in range(len(maps)):
    for y in range(3):
        if len(plant_locations[x][y][0]) != 0:
            #print(np.ptp(plant_locations[x][y], axis=1))
            plant_histograms[x][y] = np.histogram2d(plant_locations[x][y][0], plant_locations[x][y][1], bins=np.round(np.ptp(plant_locations[x][y], axis=1) / pixel_size).astype(int))
        else:
            plant_histograms[x][y] = None

ranked_plant_locations = np.full((len(rank_groups), len(maps), 3), 0, dtype=object)
for index in np.ndindex(ranked_plant_locations.shape):
    ranked_plant_locations[index] = [[], []]

for _round in round_plant_data:
    map_idx = maps_dict[_round["map"]]
    site_idx = ord(_round["plant_events"]["plant_side"].lower()) - ord("a")

    rank_idx = findRankGroup(_round["rank"])

    ranked_plant_locations[rank_idx][map_idx][site_idx][0].append(_round["plant_events"]["plant_location"]["x"])
    ranked_plant_locations[rank_idx][map_idx][site_idx][1].append(_round["plant_events"]["plant_location"]["y"])

    ranked_plant_locations[0][map_idx][site_idx][0].append(_round["plant_events"]["plant_location"]["x"])
    ranked_plant_locations[0][map_idx][site_idx][1].append(_round["plant_events"]["plant_location"]["y"])

pixel_size = 20
import matplotlib
#plt.rcParams['axes.facecolor'] = 'black'
matplotlib.rcParams['figure.figsize'] = [9, 9]

rotations = [
    [270, 0],
    [0, 0],
    [0, 0],
    [270, 270, 270],
    [90, 90],
    [0, 0]
]

all_plant_histograms = np.zeros((len(rank_groups), len(maps), 3), dtype=object)
for idx, (minR, maxR, name) in enumerate(rank_groups):
    print(name)
    plant_locations = ranked_plant_locations[idx]
    #plant_histograms = all_plant_histograms[idx]
    for x in range(len(maps)):
        for y in range(3):
            if len(plant_locations[x][y][0]) != 0:
                plant_locations[x][y] = np.array(plant_locations[x][y])
                #print(np.ptp(plant_locations[x][y], axis=1))
                all_plant_histograms[idx][x][y] = np.histogram2d(plant_locations[x][y][0], plant_locations[x][y][1], bins=np.round(np.ptp(ranked_plant_locations[0][x][y], axis=1) / pixel_size).astype(int))
            else:
                all_plant_histograms[idx][x][y] = None
    #print(idx)
    #print(all_plant_histograms == None)
    for i in range(len(maps)):
        for j in range(3):
            if all_plant_histograms[idx][i][j] is not None:
                heatmap, _, _ = all_plant_histograms[idx][i][j]
                _, xedges, yedges = all_plant_histograms[0][i][j]
                fig = plt.imshow(np.log(heatmap), extent=[yedges[0], yedges[-1], xedges[0], xedges[-1]], origin='lower')
                plt.axis('off')
                fig.axes.get_xaxis().set_visible(False)
                fig.axes.get_yaxis().set_visible(False)
                print(path)
                path = "plantlocs/" + name + "/" + maps[i] + chr(ord("A") + (j if maps[i] != "Icebox" else (1 - j))) + 'plants.png'
                plt.savefig(path, bbox_inches='tight', pad_inches = 0, transparent=True)
                plt.show()
                im = Image.open(path)
                im = im.rotate(rotations[i][j], expand=1)
                im.save(path)