import * as T from "bun:test";
import { Model } from "./lib.js";

T.test("dropOut", () => {
  const fn = Model.dropOut(0.5);
  const i = Model.uniform([10, 10], 10);
  // TODO: valueOf borks for bool eq
  T.expect(fn(i, false).eq(i).toBoolInt8()).toBe(1);
  T.expect(fn(i, true).eq(i).toBoolInt8()).toBe(0);
  T.expect(+fn(i, true).countNonzero()).toBeGreaterThan(45);
  T.expect(+fn(i, true).countNonzero()).toBeLessThan(55);
});

T.test("uniform", () => {
  const dist = Model.uniform([100, 100], 10);
  T.expect(+dist.amax()).toBeLessThan(10);
  T.expect(+dist.amin()).toBeGreaterThan(-10);
  T.expect(+dist.mean()).toBeLessThan(1);
  T.expect(+dist.mean()).toBeGreaterThan(-1);
});
