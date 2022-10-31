const fs = require("fs")
const path = require("path")

let files = fs.readdirSync("./data/full_data")
for (let file of files) {
    fs.readFile(path.join("./data/full_data", file), (err, data) => {
        let matches = JSON.parse(data);
        for (let i = 0; i < matches.length; i++) {
            for (let j = 0; j < matches[i].rounds.length; j++) {
                delete matches[i].rounds[j].player_stats
            }
        }
        fs.writeFile(path.join("./data/slim_data", file), JSON.stringify(matches), (x) => { console.log(file) })
    })
}