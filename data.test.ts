import * as T from "bun:test";
import { Data } from "./lib.js";

T.test("array", () => {
  const a = [
    [1, 2, 3],
    [4, 5, 6],
  ];
  const b = Data.toArray(Data.fromArray(a));
  T.expect(b).toEqual(a);
});

T.test("csv", () => {
  // TODO: csv test
});

T.test("disk", () => {
  const a = Data.pipeline(
    Data.fromArray([[[0, 1], [1]]]),
    Data.mapExample(
      (row) => row[0],
      (row) => row[1]
    )
  );
  Data.toDisk(a, "data/test/disk");
  const b = Data.fromDisk("data/test/disk");
  const [a1] = a();
  const [b1] = b();
  T.expect(a1[0].toFloat32Array()).toEqual(b1[0].toFloat32Array());
  T.expect(a1[1].toFloat32Array()).toEqual(b1[1].toFloat32Array());
});

// pipelines

T.test("batch", () => {
  const a = Data.pipeline(
    Data.fromArray([
      [[0, 0], [0]],
      [[1, 1], [1]],
      [[2, 2], [2]],
    ]),
    Data.mapExample(
      (row) => row[0],
      (row) => row[1]
    )
  );
  const b = Data.batch(2)(a);
  const [b1] = b();
  T.expect(b1[0].shape).toEqual([2, 2]);
  T.expect(b1[1].shape).toEqual([2, 1]);
});

T.test("cache", () => {
  const a = Data.fromArray([0, 1, 2, 3]);
  const b = Data.cache()(a);
  T.expect(Data.toArray(b)).toEqual(Data.toArray(a));
});

T.test("filter", () => {
  const a = Data.fromArray([1, 2, 3, 4]);
  const b = Data.filter((a1: number) => a1 > 2)(a);
  T.expect(Data.toArray(b)).toEqual([3, 4]);
});

T.test("log", () => {
  const a = Data.fromArray([1, 2, 3, 4]);
  const b = Data.log()(a);
  // expect no changes
  T.expect(Data.toArray(b)).toEqual(Data.toArray(a));
});

T.test("map", () => {
  const a = Data.fromArray([1, 2, 3, 4]);
  const b = Data.map(String)(a);
  T.expect(Data.toArray(b)).toEqual(["1", "2", "3", "4"]);
});

T.test("mapExample", () => {
  const a = Data.fromArray([
    [
      [1, 2],
      [3, 4],
    ],
  ]);
  const b = Data.mapExample(
    (a1: number[][]) => a1[0],
    (a1: number[][]) => a1[1]
  )(a);
  const [b1] = b();
  T.expect(b1[0].shape).toEqual([2]);
  T.expect(b1[1].shape).toEqual([2]);
  T.expect(b1[0].toFloat32Array()).toEqual(Float32Array.from([1, 2]));
  T.expect(b1[1].toFloat32Array()).toEqual(Float32Array.from([3, 4]));
});

T.test("shard", () => {
  const a = Data.fromArray([0, 1, 2, 3]);
  const [even, odd] = Data.shard()(a);
  T.expect(Data.toArray(even)).toEqual([0, 2]);
  T.expect(Data.toArray(odd)).toEqual([1, 3]);
});

// TODO: will occasionally fail by chance
T.test("shuffle", () => {
  const a = Data.fromArray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
  const b = Data.shuffle(4)(a);
  const a1 = Data.toArray(a);
  const b1 = Data.toArray(b);
  T.expect(b1.length).toBe(a1.length);
  T.expect(b1).not.toEqual(a1);
});

T.test("split", () => {
  const a = Data.fromArray([1, 2, 3, 4]);
  const [b1, b2] = Data.split(0.75)(a);
  T.expect(Data.toArray(b1)).toEqual([1, 2, 3]);
  T.expect(Data.toArray(b2)).toEqual([4]);
});
