import { Data, Model, Train } from "../../lib";

const train = Data.pipeline(
  Data.fromDisk("input/train"),
  Data.cache(), // keep dataset in memory instead of on disk
  Data.shuffle()
);

const valid = Data.pipeline(Data.fromDisk("input/valid"), Data.cache());

const model = Model.sequential(
  Model.linear(10000, 16),
  Model.relu(),
  Model.linear(16, 16),
  Model.relu(),
  Model.linear(16, 1)
);

Train.fit(model, train, valid, {
  epochs: 50,
  loss: Train.binaryCrossEntropyFromLogits(),
  metrics: [Train.accuracy],
  optimiser: Train.sgd(1e-3),
});
