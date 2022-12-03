import { test } from "bun:test";
import * as assert from "node:assert";
import * as L from "./lib";

test("confusion", () => {
  const y = L.tableToTensor([
    [0, 1, 0],
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [1, 0, 0],
  ]);
  const p = L.tableToTensor([
    [0.5, 1, 0.4],
    [0.1, 0.1, 0.3],
    [0.2, 0.1, 0.1],
    [0.00001, 0.00001, 0.99999],
    [0, 1, 0],
    [0.99, 0.99, 0.999],
  ]);
  const metric = L.confusion();
  assert.deepEqual(metric(y, p), [
    [1, 0, 1],
    [0, 2, 1],
    [0, 0, 1],
  ]);
});

test("dropOut", () => {
  const fn = L.dropOut(0.5);
  const i = L.uniform([10, 10], 10);
  // TODO: valueOf borks for bool eq
  assert(fn(i, false).eq(i).toBoolInt8() === 1);
  assert(fn(i, true).eq(i).toBoolInt8() === 0);
  assert(+fn(i, true).countNonzero() > 45 && +fn(i, true).countNonzero() < 55);
  // TODO: do the -0s need to be fixed?
  // console.log(i.valueOf(), fn(i, true).valueOf());
});

test("uniform", () => {
  const dist = L.uniform([100, 100], 10);
  assert(+dist.amax() > 9);
  assert(+dist.amin() < -9);
  assert(+dist.mean() < 1 && +dist.mean() > -1);
});
