import { Data, Model, Train } from "../lib";

const [train, valid] = Data.pipeline(
  Data.fromCsv("data/iris.csv"),
  Data.mapExample(
    (row) => row.slice(0, 4).map(Number),
    (row) => [
      row[4] === "Setosa" ? 1 : 0,
      row[4] === "Versicolor" ? 1 : 0,
      row[4] === "Virginica" ? 1 : 0,
    ]
  ),
  Data.shuffle(),
  Data.batch(10),
  Data.split(0.8)
);

const model = Model.sequential(
  Model.linear(4, 10),
  Model.relu(),
  Model.linear(10, 3),
  // TODO: softmax does not converge with change in shumai for amax gradient.
  // Adding the `.sum(axes, ctx.forward_inputs[2])` back works and includes the axes fix
  Model.softmax()
);

Train.fit(model, train, valid, {
  epochs: 300,
  loss: Train.crossEntropy(),
  metrics: [Train.accuracy, Train.confusion],
  optimiser: Train.sgd(1e-3),
});
