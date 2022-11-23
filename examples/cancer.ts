import * as SM from "@shumai/shumai";
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
  .flat();

const data = SM.tensor(Float32Array.from(rows)).reshape([rows.length / 10, 10]);

const model = L.sequential(
  L.linear(9, 10),
  L.relu(),
  L.linear(10, 10),
  L.relu(),
  L.linear(10, 1),
  L.sigmoid()
);

L.train(model, data, "0:9", 9, {
  batchSize: 10,
  epochs: 20,
  loss: L.binaryCrossEntropy(),
  optimiser: L.sgd(1e-2),
});
