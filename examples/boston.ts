import * as L from "../lib";

const text = await Bun.file("data/boston.csv").text();
const rows = text
  .split("\n")
  .slice(1, -1)
  .map((line) => L.csvLine(line).map(Number))
  .map((row) => L.example(row.slice(0, 13), row.slice(13)));

const [trainRows, validationRows] = L.split(rows, 0.75);
const data = L.dataSet(trainRows, { batchSize: 10 });
const validation = L.dataSet(validationRows, { batchSize: Infinity });

const model = L.sequential(
  L.linear(13, 20),
  L.relu(),
  L.linear(20, 20),
  L.relu(),
  L.linear(20, 1)
);

L.train(model, data, validation, {
  epochs: 20,
  loss: L.meanAbsoluteError(),
  metrics: {},
  optimiser: L.sgd(1e-5),
});
