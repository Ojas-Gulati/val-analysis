import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

from itertools import combinations

npzfile = np.load("./processed.npz")

x_agents = npzfile["x_agents"]
datapoints = len(x_agents)
train_points = int(datapoints * 0.8)

p = np.random.permutation(datapoints)
x_agents, x_agents_test = np.split(npzfile["x_agents"][p], [train_points])
x_maps, x_maps_test = np.split(npzfile["x_maps"][p], [train_points])
x_rank = npzfile["x_rank"][p]
x_rank = (x_rank - 3) / 18
x_rank = np.reshape(x_rank, (-1, 1))
x_attacking = npzfile["x_attacking"][p]
x_attacking = np.reshape(x_attacking, (-1, 1))
x_team_info, x_team_info_test = np.split(np.hstack((x_rank, x_attacking)), [train_points])

y_winning = npzfile["y_winning"][p].astype(int)

y_win_onehot = np.zeros((datapoints, 2))
y_win_onehot[np.arange(datapoints), y_winning] = 1

y_winning = y_win_onehot
y_winning, y_winning_test = np.split(y_winning, [train_points])

print(x_team_info)
print(y_winning)

agents_count = len(x_agents[0])
maps_count = len(x_maps[0])

def make_full_model():
    agents_input = keras.Input(shape=(agents_count,), name="agents")
    maps_input = keras.Input(shape=(maps_count,), name="map")
    info_input = keras.Input(shape=(2,), name="team_info")

    agents_info = keras.layers.concatenate([agents_input, info_input])
    maps_info = keras.layers.concatenate([maps_input, info_input])

    agents_1 = keras.layers.Dense(30, activation='tanh')(agents_info)
    agents_2 = keras.layers.Dense(15, activation='tanh')(agents_1)

    maps_1 = keras.layers.Dense(15, activation='tanh')(maps_info)
    maps_2 = keras.layers.Dense(15, activation='tanh')(maps_1)

    concat = keras.layers.concatenate([agents_2, maps_2])
    output = keras.layers.Dense(2, activation='tanh')(concat)
    output_softmax = keras.layers.Softmax(name="winning")(output)

    return keras.Model(
        inputs=[agents_input, maps_input, info_input],
        outputs=[output_softmax]
    )

def make_agents_model():
    agents_input = keras.Input(shape=(agents_count,), name="agents")
    maps_input = keras.Input(shape=(maps_count,), name="map")
    info_input = keras.Input(shape=(2,), name="team_info")

    agents_1 = keras.layers.Dense(64, activation='tanh')(agents_input)
    agents_2 = keras.layers.Dense(64, activation='tanh')(agents_1)

    output = keras.layers.Dense(2, activation='tanh')(agents_2)
    output_softmax = keras.layers.Softmax(name="winning")(output)

    return keras.Model(
        inputs=[agents_input, maps_input, info_input],
        outputs=[output_softmax]
    )

model = make_agents_model()
keras.utils.plot_model(model, "model.png", show_shapes=True)
print(model.summary())

model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=[keras.losses.MeanSquaredError()]
)

model.fit(
    [x_agents, x_maps, x_team_info],
    [y_winning],
    epochs=22,
    batch_size=32,
)

agents = ['Astra', 'Breach', 'Brimstone', 'Cypher', 'Jett', 'Killjoy', 'Omen', 'Phoenix', 'Raze', 'Reyna', 'Sage', 'Skye', 'Sova', 'Viper', 'Yoru']
agents_dict = {}
for x in range(len(agents)):
    agents_dict[agents[x]] = x

print(model.evaluate([x_agents_test, x_maps_test, x_team_info_test], [y_winning_test]))

results = []

onehots = np.zeros((3003, len(agents)))
maps = np.zeros((3003, 6))
info = np.zeros((3003, 2))

comboslist = []

idx = 0
for combo in combinations(agents, 5):
    comboslist.append(combo)
    for agent in combo:
        onehots[idx][agents_dict[agent]] = 1
    idx += 1

preds = model.predict([onehots, maps, info])[:,1]
print(preds)
indices = (-preds).argsort()[:10]

for index in indices:
    print(comboslist[index])
    print(preds[index])

print("-------------------------------")
indices = (preds).argsort()[:10]

for index in indices:
    print(comboslist[index])
    print(preds[index])