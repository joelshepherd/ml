import * as SM from "@shumai/shumai";
import * as FS from "node:fs";
import type * as Train from "./train.js";

/** dataset */
// lazy because the iterable protocol on its own is ambiguous
// on whether an iterable can be iterated over more than once
// this ensures it can be created a new iterable for every use
// TODO: consider moving away from generator functions and
//       building reiterables manually
export type Dataset<T> = () => Iterable<T>;

/** create dataset from array-like */
export const fromArray =
  <T>(data: ArrayLike<T>): Dataset<T> =>
  () =>
    Array.from(data).values();

/** export dataset to array */
export const toArray = <T>(data: Dataset<T>): Array<T> => Array.from(data());

/** import dataset from csv */
// TODO: stop loading entire file into memory
// TODO: header row options
// TODO: data types options
export const fromCsv = (path: string): Dataset<string[]> => {
  const data = FS.readFileSync(path, { encoding: "utf-8" })
    .split("\n")
    .slice(1)
    .map(csvLine);
  return () => data.values();
};

/** import example dataset from disk */
export const fromDisk = (path: string): Dataset<Train.Example> =>
  function* () {
    for (const i of naturals()) {
      const prefix = `${path}/${i}`;
      if (!FS.existsSync(prefix + "-0")) break;
      const x = SM.tensor(prefix + "-0");
      const y = SM.tensor(prefix + "-1");
      yield [x, y];
    }
  };

/** export example dataset to disk */
export const toDisk = (data: Dataset<Train.Example>, path: string): void => {
  let i = 0;
  for (const [x, y] of data()) {
    x.save(`${path}/${i}-0`);
    y.save(`${path}/${i}-1`);
    i += 1;
  }
};

// data pipelines

/** create data pipeline */
export const pipeline = pipe;

/** combine example dataset into batches */
export const batch =
  (size: number) =>
  (data: Dataset<Train.Example>): Dataset<Train.Example> =>
    function* () {
      let batchX: SM.Tensor | null = null;
      let batchY: SM.Tensor | null = null;
      let i = 0;
      for (const [x, y] of data()) {
        batchX = batchX ? SM.concat([batchX, x], -2) : x;
        batchY = batchY ? SM.concat([batchY, y], -2) : y;
        i += 1;
        if (i === size) {
          yield [batchX, batchY];
          batchX = null;
          batchY = null;
          i = 0;
        }
      }
      // yield remainder
      if (batchX && batchY) yield [batchX, batchY];
    };

/** cache dataset into memory */
export const cache =
  () =>
  <T>(data: Dataset<T>): Dataset<T> => {
    const buffer = Array.from(data());
    return () => buffer.values();
  };

/** filter dataset */
export const filter =
  <T>(fn: (input: T) => boolean) =>
  (data: Dataset<T>): Dataset<T> =>
    function* () {
      for (const row of data()) if (fn(row)) yield row;
    };

/** log dataset */
export const log =
  () =>
  <T>(data: Dataset<T>): Dataset<T> =>
    function* () {
      let i = 0;
      for (const row of data()) {
        console.log(`row(${i}):`, row);
        yield row;
        i += 1;
      }
    };

/** map dataset */
export const map =
  <I, O>(fn: (input: I) => O) =>
  (data: Dataset<I>): Dataset<O> =>
    function* () {
      for (const row of data()) yield fn(row);
    };

/** map dataset to examples */
export const mapExample =
  <T>(mapFeature: (input: T) => number[], mapLabel: (input: T) => number[]) =>
  (data: Dataset<T>): Dataset<Train.Example> =>
    function* () {
      for (const row of data()) {
        yield [
          SM.tensor(Float32Array.from(mapFeature(row))),
          SM.tensor(Float32Array.from(mapLabel(row))),
        ];
      }
    };

/** map tuple of number arrays to examples */
// TODO: consider removal
export const mapExampleTuple =
  () =>
  (data: Dataset<[number[], number[]]>): Dataset<Train.Example> =>
    function* () {
      for (const [feature, label] of data()) {
        yield [
          SM.tensor(Float32Array.from(feature)),
          SM.tensor(Float32Array.from(label)),
        ];
      }
    };

/** shard dataset */
// TODO: arbitrary splits
// TODO: make efficient - stop wasting reads from dataset
// TODO: overlap between this and split?
export const shard =
  () =>
  <T>(data: Dataset<T>): [Dataset<T>, Dataset<T>] => {
    function* split(first: boolean) {
      for (const example of data()) {
        if (first) yield example;
        first = !first;
      }
    }
    return [() => split(true), () => split(false)];
  };

/** shuffle dataset */
export const shuffle =
  (window: number = Infinity) =>
  <T>(data: Dataset<T>): Dataset<T> =>
    function* () {
      let buffer: T[] = [];
      for (const example of data()) {
        buffer.push(example);
        if (buffer.length === window) {
          yield* SM.util.shuffle(buffer);
          buffer = [];
        }
      }
      yield* SM.util.shuffle(buffer);
    };

/** split dataset */
// TODO: memory efficient splits
export const split =
  (ratio: number) =>
  <T>(data: Dataset<T>): [Dataset<T>, Dataset<T>] => {
    const buffer = Array.from(data());
    const index = Math.round(buffer.length * ratio);
    return [
      () => buffer.slice(0, index).values(),
      () => buffer.slice(index).values(),
    ];
    // TODO: for memory efficient splits in buffers
    // TODO: only works if 0.5<=ratio<=1
    // const split = ratio / 1 - ratio;
    // const bufferSize = split / ratio;
  };

// utils

function* naturals(): Generator<number> {
  let i = 0;
  while (true) {
    yield i;
    i += 1;
  }
}

// TODO: escaping quotes
const csvLine = (line: string): string[] => {
  const cells = [];
  let cell = "";
  let quoting = false;
  for (let letter of line) {
    if (letter === '"') {
      quoting = !quoting;
    } else if (!quoting && letter === ",") {
      cells.push(cell);
      cell = "";
    } else {
      cell += letter;
    }
  }
  cells.push(cell);
  return cells;
};

function pipe<A>(a: A): A;
function pipe<A, B>(a: A, ab: (a: A) => B): B;
function pipe<A, B, C>(a: A, ab: (a: A) => B, bc: (b: B) => C): C;
function pipe<A, B, C, D>(
  a: A,
  ab: (a: A) => B,
  bc: (b: B) => C,
  cd: (c: C) => D
): D;
function pipe<A, B, C, D, E>(
  a: A,
  ab: (a: A) => B,
  bc: (b: B) => C,
  cd: (c: C) => D,
  de: (d: D) => E
): E;
function pipe<A, B, C, D, E, F>(
  a: A,
  ab: (a: A) => B,
  bc: (b: B) => C,
  cd: (c: C) => D,
  de: (d: D) => E,
  ef: (e: E) => F
): F;
function pipe<A, B, C, D, E, F, G>(
  a: A,
  ab: (a: A) => B,
  bc: (b: B) => C,
  cd: (c: C) => D,
  de: (d: D) => E,
  ef: (e: E) => F,
  fg: (f: F) => G
): G;
function pipe<A, B, C, D, E, F, G, H>(
  a: A,
  ab: (a: A) => B,
  bc: (b: B) => C,
  cd: (c: C) => D,
  de: (d: D) => E,
  ef: (e: E) => F,
  fg: (f: F) => G,
  gh: (g: G) => H
): H;
function pipe<A, B, C, D, E, F, G, H, I>(
  a: A,
  ab: (a: A) => B,
  bc: (b: B) => C,
  cd: (c: C) => D,
  de: (d: D) => E,
  ef: (e: E) => F,
  fg: (f: F) => G,
  gh: (g: G) => H,
  hi: (h: H) => I
): I;
function pipe<A, B, C, D, E, F, G, H, I, J>(
  a: A,
  ab: (a: A) => B,
  bc: (b: B) => C,
  cd: (c: C) => D,
  de: (d: D) => E,
  ef: (e: E) => F,
  fg: (f: F) => G,
  gh: (g: G) => H,
  hi: (h: H) => I,
  ij: (i: I) => J
): J;
function pipe<A, B, C, D, E, F, G, H, I, J, K>(
  a: A,
  ab: (a: A) => B,
  bc: (b: B) => C,
  cd: (c: C) => D,
  de: (d: D) => E,
  ef: (e: E) => F,
  fg: (f: F) => G,
  gh: (g: G) => H,
  hi: (h: H) => I,
  ij: (i: I) => J,
  jk: (j: J) => K
): K;
function pipe<A, B, C, D, E, F, G, H, I, J, K, L>(
  a: A,
  ab: (a: A) => B,
  bc: (b: B) => C,
  cd: (c: C) => D,
  de: (d: D) => E,
  ef: (e: E) => F,
  fg: (f: F) => G,
  gh: (g: G) => H,
  hi: (h: H) => I,
  ij: (i: I) => J,
  jk: (j: J) => K,
  kl: (k: K) => L
): L;
function pipe<A, B, C, D, E, F, G, H, I, J, K, L, M>(
  a: A,
  ab: (a: A) => B,
  bc: (b: B) => C,
  cd: (c: C) => D,
  de: (d: D) => E,
  ef: (e: E) => F,
  fg: (f: F) => G,
  gh: (g: G) => H,
  hi: (h: H) => I,
  ij: (i: I) => J,
  jk: (j: J) => K,
  kl: (k: K) => L,
  lm: (l: L) => M
): M;
function pipe<A, B, C, D, E, F, G, H, I, J, K, L, M, N>(
  a: A,
  ab: (a: A) => B,
  bc: (b: B) => C,
  cd: (c: C) => D,
  de: (d: D) => E,
  ef: (e: E) => F,
  fg: (f: F) => G,
  gh: (g: G) => H,
  hi: (h: H) => I,
  ij: (i: I) => J,
  jk: (j: J) => K,
  kl: (k: K) => L,
  lm: (l: L) => M,
  mn: (m: M) => N
): N;
function pipe<A, B, C, D, E, F, G, H, I, J, K, L, M, N, O>(
  a: A,
  ab: (a: A) => B,
  bc: (b: B) => C,
  cd: (c: C) => D,
  de: (d: D) => E,
  ef: (e: E) => F,
  fg: (f: F) => G,
  gh: (g: G) => H,
  hi: (h: H) => I,
  ij: (i: I) => J,
  jk: (j: J) => K,
  kl: (k: K) => L,
  lm: (l: L) => M,
  mn: (m: M) => N,
  no: (n: N) => O
): O;
function pipe<A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P>(
  a: A,
  ab: (a: A) => B,
  bc: (b: B) => C,
  cd: (c: C) => D,
  de: (d: D) => E,
  ef: (e: E) => F,
  fg: (f: F) => G,
  gh: (g: G) => H,
  hi: (h: H) => I,
  ij: (i: I) => J,
  jk: (j: J) => K,
  kl: (k: K) => L,
  lm: (l: L) => M,
  mn: (m: M) => N,
  no: (n: N) => O,
  op: (o: O) => P
): P;
function pipe<A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q>(
  a: A,
  ab: (a: A) => B,
  bc: (b: B) => C,
  cd: (c: C) => D,
  de: (d: D) => E,
  ef: (e: E) => F,
  fg: (f: F) => G,
  gh: (g: G) => H,
  hi: (h: H) => I,
  ij: (i: I) => J,
  jk: (j: J) => K,
  kl: (k: K) => L,
  lm: (l: L) => M,
  mn: (m: M) => N,
  no: (n: N) => O,
  op: (o: O) => P,
  pq: (p: P) => Q
): Q;
function pipe<A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R>(
  a: A,
  ab: (a: A) => B,
  bc: (b: B) => C,
  cd: (c: C) => D,
  de: (d: D) => E,
  ef: (e: E) => F,
  fg: (f: F) => G,
  gh: (g: G) => H,
  hi: (h: H) => I,
  ij: (i: I) => J,
  jk: (j: J) => K,
  kl: (k: K) => L,
  lm: (l: L) => M,
  mn: (m: M) => N,
  no: (n: N) => O,
  op: (o: O) => P,
  pq: (p: P) => Q,
  qr: (q: Q) => R
): R;
function pipe<A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S>(
  a: A,
  ab: (a: A) => B,
  bc: (b: B) => C,
  cd: (c: C) => D,
  de: (d: D) => E,
  ef: (e: E) => F,
  fg: (f: F) => G,
  gh: (g: G) => H,
  hi: (h: H) => I,
  ij: (i: I) => J,
  jk: (j: J) => K,
  kl: (k: K) => L,
  lm: (l: L) => M,
  mn: (m: M) => N,
  no: (n: N) => O,
  op: (o: O) => P,
  pq: (p: P) => Q,
  qr: (q: Q) => R,
  rs: (r: R) => S
): S;
function pipe<A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T>(
  a: A,
  ab: (a: A) => B,
  bc: (b: B) => C,
  cd: (c: C) => D,
  de: (d: D) => E,
  ef: (e: E) => F,
  fg: (f: F) => G,
  gh: (g: G) => H,
  hi: (h: H) => I,
  ij: (i: I) => J,
  jk: (j: J) => K,
  kl: (k: K) => L,
  lm: (l: L) => M,
  mn: (m: M) => N,
  no: (n: N) => O,
  op: (o: O) => P,
  pq: (p: P) => Q,
  qr: (q: Q) => R,
  rs: (r: R) => S,
  st: (s: S) => T
): T;
function pipe(
  a: unknown,
  ab?: Function,
  bc?: Function,
  cd?: Function,
  de?: Function,
  ef?: Function,
  fg?: Function,
  gh?: Function,
  hi?: Function
): unknown {
  switch (arguments.length) {
    case 1:
      return a;
    case 2:
      return ab!(a);
    case 3:
      return bc!(ab!(a));
    case 4:
      return cd!(bc!(ab!(a)));
    case 5:
      return de!(cd!(bc!(ab!(a))));
    case 6:
      return ef!(de!(cd!(bc!(ab!(a)))));
    case 7:
      return fg!(ef!(de!(cd!(bc!(ab!(a))))));
    case 8:
      return gh!(fg!(ef!(de!(cd!(bc!(ab!(a)))))));
    case 9:
      return hi!(gh!(fg!(ef!(de!(cd!(bc!(ab!(a))))))));
    default: {
      let ret = arguments[0];
      for (let i = 1; i < arguments.length; i++) {
        ret = arguments[i](ret);
      }
      return ret;
    }
  }
}
