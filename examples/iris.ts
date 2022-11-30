import * as SM from "@shumai/shumai";
import * as Lo from "lodash";
import * as L from "../lib";

// Data
type Raw = [
  string,
  string,
  string,
  string,
  "Setosa" | "Versicolor" | "Virginica"
];
type Row = [number, number, number, number, 0 | 1, 0 | 1, 0 | 1];

const text = await Bun.file("./data/iris.csv").text();
const rows: Row[] = text
  .split("\n")
  .slice(1, -1)
  .map((line) => {
    const raw = L.csvLine(line) as Raw;
    return [
      Number(raw[0]),
      Number(raw[1]),
      Number(raw[2]),
      Number(raw[3]),
      raw[4] === "Setosa" ? 1 : 0,
      raw[4] === "Versicolor" ? 1 : 0,
      raw[4] === "Virginica" ? 1 : 0,
    ];
  });

const data = L.dataSet(L.tableToTensor(SM.util.shuffle(rows)), {
  xIndex: "0:4",
  yIndex: "4:7",
  batchSize: 10,
  validationPortion: 0.25,
});

const model = L.sequential(
  L.linear(4, 10),
  L.relu(),
  L.linear(10, 10),
  L.relu(),
  L.linear(10, 3),
  L.softmax()
);

L.train(model, data, {
  epochs: 100,
  loss: L.crossEntropy(),
  metrics: {
    accuracy: L.accuracy(),
    // precision: L.precision(),
    // f1: L.f1(),
  },
  optimiser: L.sgd(1e-3),
});

// test
const X = data[1][1];
const Y = data[1][0];
const P = model(X);

Lo.take(
  SM.util.shuffle(
    Lo.zip(Lo.chunk(Y.toFloat32Array(), 3), Lo.chunk(P.toFloat32Array(), 3))
  ),
  10
).forEach(([y, p]) => console.log(y, "v", p));
