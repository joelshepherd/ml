import * as SM from "@shumai/shumai";
import * as L from "../lib";

const text = await Bun.file("data/cancer.csv").text();
const data = text
  .split("\n")
  .slice(1, -1)
  .map((line) =>
    line
      .split(",")
      .slice(1)
      .map((cell, i) => (i === 9 ? (cell === "2" ? 0 : 1) : Number(cell)))
  )
  .filter((row) => !row.some(Number.isNaN))
  .map<[SM.Tensor, SM.Tensor]>((row) => [
    SM.scalar(row.pop()),
    SM.tensor(Float32Array.from(row)),
  ]);

const model = L.sequential(
  L.linear(9, 20),
  L.relu(),
  L.dropOut(20, 0.5),
  L.linear(20, 1),
  L.sigmoid()
);

L.train(model, data, {
  batchSize: 1,
  epochs: 10,
  loss: L.binaryCrossEntropy(),
  optimiser: L.sgd(1e3),
});
