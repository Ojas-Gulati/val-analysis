const fs = require("fs")

let mergedData = [];

const matchesFolder = "../data/matches";

let files = fs.readdirSync("../data/matches");

for (let file of files) {
    mergedData.push(...JSON.parse(fs.readFileSync(matchesFolder + "/" + file)));
}

console.log("Merged " + mergedData.length + " records")
fs.writeFileSync("fulldata.json", JSON.stringify(mergedData));