import { Data, Model, Train } from "../lib.js";

const [train, valid] = Data.pipeline(
  Data.fromCsv("data/boston.csv"),
  Data.mapExample(
    (row) => row.slice(0, 13).map(Number),
    (row) => row.slice(13).map(Number)
  ),
  Data.shuffle(),
  Data.batch(10),
  Data.split(0.8)
);

const model = Model.sequential(
  Model.linear(13, 20),
  Model.relu(),
  Model.linear(20, 20),
  Model.relu(),
  Model.linear(20, 1)
);

Train.fit(model, train, valid, {
  epochs: 100,
  loss: Train.meanAbsoluteError(),
  optimiser: Train.sgd(1e-5),
});
