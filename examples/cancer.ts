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
  .filter((row) => !row.some(Number.isNaN));

const data = L.dataSet(L.tableToTensor(rows), {
  xIndex: "0:9",
  yIndex: 9,
  batchSize: 10,
  validationPortion: 0.25,
});

const model = L.sequential(
  L.linear(9, 10),
  L.relu(),
  L.linear(10, 10),
  L.relu(),
  L.linear(10, 1),
  L.sigmoid()
);

L.train(model, data, {
  epochs: 10,
  loss: L.binaryCrossEntropy(),
  metrics: {
    accuracy: L.accuracy(),
    precision: L.precision(),
    f1: L.f1(),
  },
  optimiser: L.sgd(1e-1),
});
