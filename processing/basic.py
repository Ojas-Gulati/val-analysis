import json
import numpy as np
import matplotlib.pyplot as plt

with open('./fulldata.json') as f:
    data = json.load(f)

# process this into an numpy array
# step 0: seperate red & blue (and also get a list of agents and maps)
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
print(agents)

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

no_ranks = 19
min_rank = 3

fig, ax = plt.subplots(3, 5)
fig.tight_layout()

for i in range(3):
    for j in range(5):
        agent_no = (i * 5) + j
        find_pickrate_of = agents[agent_no]
        peek_agent = agents_dict[find_pickrate_of] # Sage
        wins = np.zeros((no_ranks,))
        losses = np.zeros((no_ranks,))
        total = np.zeros((no_ranks,))
        for x in range(datasize):
            curr_rank = int(np.floor(x_rank[x]) - min_rank)
            total[curr_rank] += 1
            if (x_agents[x][peek_agent] == 1):
                # sage was played here
                if (y_winning[x] == 1):
                    wins[curr_rank] += 1
                else:
                    losses[curr_rank] += 1

        pickrate = (wins + losses) / total

        ax[i, j].title.set_text(find_pickrate_of)
        ax[i, j].bar(np.arange(no_ranks), pickrate)
plt.show()