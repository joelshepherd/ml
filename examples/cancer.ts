import * as L from "../lib";

const text = await Bun.file("data/cancer.csv").text();
const rows = text
  .split("\n")
  .slice(1, -1)
  .map((line) =>
    line
      .split(",")
      .slice(1)
      .map((cell, i) => (i === 9 ? (cell === "2" ? 0 : 1) : Number(cell)))
  )
  .filter((row) => !row.some(Number.isNaN))
  .map((row) => L.example(row.slice(0, 9), row.slice(9)));

const [trainRows, validationRows] = L.split(rows, 0.75);

const data = L.dataSet(trainRows, { batchSize: 10 });
const validation = L.dataSet(validationRows, { batchSize: Infinity });

const model = L.sequential(
  L.linear(9, 10),
  L.relu(),
  L.linear(10, 10),
  L.relu(),
  L.linear(10, 1),
  L.sigmoid()
);

L.train(model, data, validation, {
  epochs: 10,
  loss: L.binaryCrossEntropy(),
  metrics: {
    accuracy: L.accuracy(),
    precision: L.precision(),
    f1: L.f1(),
  },
  optimiser: L.sgd(1e-2),
});
