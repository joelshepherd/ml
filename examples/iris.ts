import * as L from "../lib";

// Data
type Row = [
  string,
  string,
  string,
  string,
  "Setosa" | "Versicolor" | "Virginica"
];

const text = await Bun.file("./data/iris.csv").text();
const rows = text
  .split("\n")
  .slice(1, -1)
  .map((line) => {
    const row = L.csvLine(line) as Row;
    return L.example(row.slice(0, 4).map(Number), [
      row[4] === "Setosa" ? 1 : 0,
      row[4] === "Versicolor" ? 1 : 0,
      row[4] === "Virginica" ? 1 : 0,
    ]);
  });

const [trainRows, validationRows] = L.split(rows, 0.8);
const train = L.dataSet(trainRows, { batchSize: 5 });
const validation = L.dataSet(validationRows, { batchSize: Infinity });

const model = L.sequential(
  L.linear(4, 10),
  L.relu(),
  L.linear(10, 3),
  L.softmax()
);

L.train(model, train, validation, {
  epochs: 400,
  loss: L.crossEntropy(),
  metrics: {
    accuracy: L.accuracy(),
    confusion: L.confusion(),
  },
  optimiser: L.sgd(1e-3),
});
