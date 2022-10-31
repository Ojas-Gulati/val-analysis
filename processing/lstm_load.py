import json
import numpy as np
import matplotlib.pyplot as plt

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

# np.savez("./processed.npz", x_agents=x_agents, x_maps=x_maps, x_rank=x_rank, x_attacking=x_attacking, y_winning=y_winning)
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

# print(len(round_plant_data))
# print(round_plant_data[0])
# print(np.min(rankslist))
# print(np.max(rankslist))
# print(np.median(rankslist))

# round_plant_data has the goods
# first of all, lets split our data by map
round_plant_data_by_map = np.zeros((len(maps),), dtype=object)
for x in range(len(maps)):
    round_plant_data_by_map[x] = []
for _round in round_plant_data:
    round_plant_data_by_map[maps_dict[_round["map"]]].append(_round)

for x in range(len(maps)):
    print(maps[x] + " " + str(len(round_plant_data_by_map[x])))

minmax_coords = np.zeros((len(maps), 2, 2))
minmax_coords[:, :, 0] = np.inf
minmax_coords[:, :, 1] = -np.inf

axes = ["x", "y"]

import codecs, json

for map_idx in range(len(maps)):
    for _round in round_plant_data_by_map[map_idx]:
        for locobj in _round["plant_events"]["player_locations_on_plant"]:
            for x in range(2):
                minmax_coords[map_idx][x][0] = min(minmax_coords[map_idx][x][0], locobj["location"][axes[x]])
                minmax_coords[map_idx][x][1] = max(minmax_coords[map_idx][x][1], locobj["location"][axes[x]])

print(minmax_coords)
file_path = "./minmaxcoords.json" ## your path variable
json.dump(minmax_coords.tolist(), codecs.open(file_path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4) ### this saves the array in .json format

# for now we're only going to do one map
# attacking (bomb planting team) will be 0, defending (bomb defusing team) will be 1
import tensorflow as tf
from tensorflow import keras

def minmax_normalise(no, minmax):
    return (no - minmax[0]) / (minmax[1] - minmax[0])

checkpoint_names = [
    "mymodelFIN_0_28",
    "mymodelFIN_1_36",
    "mymodelFIN_2_29",
    "mymodelFIN_3_34",
    "mymodelFIN_4_31",
    "mymodelFIN_5_21"
]
for map_idx in range(len(maps)):
    normalisation_data = minmax_coords[map_idx]

    x_plantLocations = np.zeros((len(round_plant_data_by_map[map_idx]), 2))
    x_playerLocations = np.zeros((len(round_plant_data_by_map[map_idx]), 10, 2))
    x_playerTeams = np.zeros((len(round_plant_data_by_map[map_idx]), 10, 1)) # -1 if attacking player, 1 if defending
    y_winners = np.zeros((len(round_plant_data_by_map[map_idx]),)) # -1 if attacking wins, 1 if defending
    for idx, _round in enumerate(round_plant_data_by_map[map_idx]):
        planter = _round["plant_events"]["planted_by"]["team"].lower()

        x_plantLocations[idx][0] = minmax_normalise(_round["plant_events"]["plant_location"]["x"], normalisation_data[0])
        x_plantLocations[idx][1] = minmax_normalise(_round["plant_events"]["plant_location"]["y"], normalisation_data[1])

        player_locs = sorted(_round["plant_events"]["player_locations_on_plant"], key=lambda k: k["location"]["y"])
        for pl_idx, location in enumerate(player_locs):
            # we need to order these by y later
            x_playerLocations[idx][pl_idx][0] = minmax_normalise(location["location"]["x"], normalisation_data[0])
            x_playerLocations[idx][pl_idx][1] = minmax_normalise(location["location"]["y"], normalisation_data[1])
            x_playerTeams[idx][pl_idx][0] = -1 if location["player_team"].lower() == planter else 1

        y_winners[idx] = 0 if _round["winning_team"].lower() == planter else 1

    # print(x_plantLocations[:2])
    # print(x_playerLocations[:2])
    # print(x_playerTeams[:2])
    # print(y_winners[:2])

    def makeLSTMmodel(): #AR4
        plantLocations_input_um = keras.Input(shape=(2,), name="plantLocations")
        playerLocations_input_um = keras.Input(shape=(None, 2), name="playerLocations")
        playerTeams_input_um = keras.Input(shape=(None, 1), name="playerTeams")
        
        mask = keras.layers.Masking()
        plantLocations_input = mask(plantLocations_input_um)
        playerLocations_input = mask(playerLocations_input_um)
        playerTeams_input = mask(playerTeams_input_um)

        denseLayer1 = keras.layers.Dense(16, activation='relu')
        denseLayer2 = keras.layers.Dense(8, activation='relu')

        def locationEncode(x):
            return denseLayer2(denseLayer1(x))

        plant_location_encode = locationEncode(plantLocations_input)
        players_location_encode = locationEncode(playerLocations_input)
        players_and_teams = keras.layers.Concatenate()([players_location_encode, playerTeams_input])

        lstm_size = 64
        lstm_layer = keras.layers.LSTM(lstm_size)(players_and_teams, initial_state=[keras.layers.Dense(lstm_size)(plant_location_encode), keras.layers.Dense(lstm_size)(plant_location_encode)])
        # lstm_layer_2 = keras.layers.LSTM(64)(lstm_layer)
        fin2_layer = keras.layers.Dense(16, activation='relu')(lstm_layer)
        final_layer = keras.layers.Dense(1, activation='sigmoid')(fin2_layer)

        return keras.Model(inputs=[plantLocations_input_um, playerLocations_input_um, playerTeams_input_um], outputs=final_layer)

    model = makeLSTMmodel()
    keras.utils.plot_model(model, "model.png", show_shapes=True)
    # print(model.summary())

    import datetime
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    # now we can train! :)
    datapoints = len(x_plantLocations)
    p = np.random.permutation(datapoints)
    train_points = int(datapoints * 0.8)

    x_plantLocations, x_plantLocations_test = np.split(x_plantLocations[p], [train_points])
    x_playerLocations, x_playerLocations_test = np.split(x_playerLocations[p], [train_points])
    x_playerTeams, x_playerTeams_test = np.split(x_playerTeams[p], [train_points])
    y_winners, y_winners_test = np.split(y_winners[p], [train_points])

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=[keras.losses.BinaryCrossentropy()],
        metrics=['accuracy']
    )

    loss, acc = model.evaluate([np.array(x_plantLocations_test, dtype=np.float), np.array(x_playerLocations_test, dtype=np.float), np.array(x_playerTeams_test, dtype=np.float)], np.array(y_winners_test, dtype=np.float), verbose=2)
    print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))

    model = tf.keras.models.load_model("./modelsaves/" + checkpoint_names[map_idx])
    loss, acc = model.evaluate([np.array(x_plantLocations_test, dtype=np.float), np.array(x_playerLocations_test, dtype=np.float), np.array(x_playerTeams_test, dtype=np.float)], np.array(y_winners_test, dtype=np.float), verbose=2)
    print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

    model.save(maps[map_idx] + "model.h5")

    # log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    # checkpoint_callback = keras.callbacks.ModelCheckpoint(
    #         # Path where to save the model
    #         # The two parameters below mean that we will overwrite
    #         # the current checkpoint if and only if
    #         # the `val_loss` score has improved.
    #         # The saved model name will include the current epoch.
    #         filepath="./modelsaves/mymodelFIN_" + str(map_idx) + "_{epoch}",
    #         save_best_only=True,  # Only save a model if `val_loss` has improved.
    #         monitor="val_loss",
    #         verbose=1,
    #     )