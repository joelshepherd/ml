import { Data, Model, Train } from "../lib.js";

const [train, valid] = Data.pipeline(
  Data.fromCsv("data/cancer.csv"),
  Data.map((row) => row.map(Number)),
  // remove rows with missing data
  Data.filter((row) => !row.some(Number.isNaN)),
  Data.mapExample(
    (row) => row.slice(1, 10),
    (row) => [row[10] === 2 ? 0 : 1]
  ),
  Data.shuffle(),
  Data.batch(10),
  Data.split(0.8)
);

const model = Model.sequential(
  Model.linear(9, 10),
  Model.relu(),
  Model.linear(10, 10),
  Model.relu(),
  Model.linear(10, 1),
  Model.sigmoid()
);

Train.fit(model, train, valid, {
  epochs: 100,
  loss: Train.binaryCrossEntropy(),
  metrics: [Train.accuracy], // TODO: precision, recall, f1, confusion
  optimiser: Train.sgd(1e-3),
});
