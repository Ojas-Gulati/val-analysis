const fetch = require('node-fetch');
const fs = require('fs');
const { time } = require('console');

const location = "eu";

const processed_data_chunk_size = 100;

const processedPUUIDsPath = "./data/puuids.json"
const queuedPUUIDsPath = "./data/queued_puuids.json"
const matchIDsPath = "./data/matchids.json"

let processedPUUIDs = JSON.parse(fs.readFileSync(processedPUUIDsPath));
let queuedPUUIDs = JSON.parse(fs.readFileSync(queuedPUUIDsPath));
let matchIDs = JSON.parse(fs.readFileSync(matchIDsPath));
let jobs = [];
let processed_data = [];
let slim_data = [];

async function fetchMatches(puuid) {
    console.log(puuid, processed_data.length);
    let json = await fetch('https://api.henrikdev.xyz/valorant/v3/by-puuid/matches/eu/' + puuid).then(res => res.json())
    if (processedPUUIDs[puuid] === undefined) {
        let puuids = [];
        try {
            if (json.status != '200') { throw Error("Request failed: error code " + json.status) }
            // alles gut
            // each match produces 2 datapoints: a winning one for the winning team comp
            // and a losing one for the losing team comp
            for (let match of json.data) {
                if (match.metadata == undefined) {
                    throw Error("undefined metadata?");
                }
                let matchid = match.metadata.matchid;
                if (matchIDs[matchid] === undefined) {
                    if (match.metadata.mode == "Competitive") {
                        let data = {
                            "matchid": matchid,
                            "red": {
                                agents: [],
                                win: false,
                                map: match.metadata.map,
                                avg_rank: 0
                            },
                            "blue": {
                                agents: [],
                                win: false,
                                map: match.metadata.map,
                                avg_rank: 0
                            }
                        }
                        const teams = ["red", "blue"];
                        for (let team of teams) {
                            let rank = 0;
                            let rankedPlayers = 0;
                            for (let player of match.players[team]) {
                                puuids.push(player.puuid);
                                data[team].agents.push(player.character);
                                if (player.currenttier != 0) {
                                    rank += player.currenttier;
                                    rankedPlayers += 1;
                                }
                            }
                            data[team].win = match.teams[team].has_won;
                            data[team].avg_rank = rank / rankedPlayers;
                        }
                        processed_data.push(data);
                        matchIDs[matchid] = true;
                    }
                }

                for (let j = 0; j < match.rounds.length; j++) {
                    delete match.rounds[j].player_stats
                }
            }
            slim_data.push(...json.data);
            // see if we can get any more puuids
            for (let newPUUID of puuids) {
                if (queuedPUUIDs[newPUUID] === undefined) {
                    queuedPUUIDs[newPUUID] = true;
                    jobs.push([fetchMatches, [newPUUID]]);
                }
            }
            processedPUUIDs[puuid] = true;
            queuedPUUIDs[puuid] = undefined;
        }
        catch (e) {
            console.log(e);
            jobs.unshift([fetchMatches, [puuid]]);
            throw e;
        }
    }
}

// initialise jobs
for (let key in queuedPUUIDs) {
    if (queuedPUUIDs[key] === true && processedPUUIDs[key] === undefined) {
        jobs.push([fetchMatches, [key]])
    }
}

async function runner() {
    let timeout = 200;
    try {
        if (processed_data.length >= processed_data_chunk_size) {
            // do saving things
            fs.writeFileSync(processedPUUIDsPath, JSON.stringify(processedPUUIDs));
            fs.writeFileSync(queuedPUUIDsPath, JSON.stringify(queuedPUUIDs));
            fs.writeFileSync(matchIDsPath, JSON.stringify(matchIDs));

            const fullDataFiles = fs.readdirSync('./data/slim_data').length;
            fs.writeFileSync("./data/slim_data/" + fullDataFiles + ".json", JSON.stringify(slim_data));
            const matchesFiles = fs.readdirSync('./data/matches').length;
            fs.writeFileSync("./data/matches/" + matchesFiles + ".json", JSON.stringify(processed_data));
            processed_data = [];
            slim_data = [];
            console.log("saved");
        }
        else {
            if (jobs.length > 0) {
                let [func, args] = jobs[jobs.length - 1];
                try {
                    await func(...args);
                }
                catch (e) {
                    timeout = 5000;
                }
                jobs.pop();
            }
        }
    }
    catch (e) {
        console.log(e)
    }
    setTimeout(runner, timeout)
}
runner();