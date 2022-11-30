import { test } from "bun:test";
import * as assert from "node:assert";
import * as L from "./lib";

test("dropOut", () => {
  const fn = L.dropOut(0.5);
  const i = L.uniform([10, 10], 10);
  // TODO: valueOf borks for bool eq
  assert(fn(i, false).eq(i).toBoolInt8() === 1);
  assert(fn(i, true).eq(i).toBoolInt8() === 0);
  assert(+fn(i, true).countNonzero() > 45 && +fn(i, true).countNonzero() < 55);
  console.log(i.valueOf(), fn(i, true).valueOf());
});

test("uniform", () => {
  const dist = L.uniform([100, 100], 10);
  assert(+dist.amax() > 9);
  assert(+dist.amin() < -9);
  assert(+dist.mean() < 1 && +dist.mean() > -1);
});
