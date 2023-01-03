import * as SM from "@shumai/shumai";

// layers

/** model layer */
export type Layer = (input: SM.Tensor, train?: boolean) => SM.Tensor;

/** combine layers sequentially into a model */
export const sequential =
  (...layers: Layer[]): Layer =>
  (input, train = false) =>
    layers.reduce((input, layer) => layer(input, train), input);

/** linear layer */
export const linear = (
  inSize: number,
  outSize: number,
  initialiser = he()
): Layer => {
  const W = initialiser([inSize, outSize]).requireGrad();
  return (input) => input.matmul(W);
};

/** bias layer */
export const bias = (size: number, initialiser = he()): Layer => {
  // TODO: biases seems to adjust the number of inputs when optimising
  const B = initialiser([size]).requireGrad();
  return (input) => input.add(B);
};

/** drop-out regularisation layer */
export const dropOut = (keepProb: number): Layer => {
  const p = SM.scalar(keepProb);
  return (input, train) =>
    train ? input.mul(SM.rand(input.shape).lessThan(p)).div(p) : input;
};

// activation fns

/** relu activation */
export const relu = (): Layer => {
  const t = SM.scalar(0);
  return (input) => input.maximum(t);
};

/**
 * sigmoid activation
 * sm has a native implementation
 */
export const sigmoid = (): Layer => SM.sigmoid;

/** softmax activation */
export const softmax = (): Layer => (input) => {
  // for numerical stability, reduce inputs below 0
  // see: https://cs231n.github.io/linear-classify/#softmax
  const e = input.sub(input.amax([-1], true)).exp();
  return e.div(e.sum([-1], true));
};

// initialisers

/** layer initialiser */
type Initialiser = (dimensions: number[]) => SM.Tensor;

/**
 * xavier initialiser
 * use with tahn
 */
export const xavier =
  (gain = 1): Initialiser =>
  (dimensions) =>
    uniform(dimensions, gain * Math.sqrt(6 / (dimensions[0] + dimensions[1])));

/**
 * he initialiser
 * use with relu
 */
export const he = (): Initialiser => (dimensions) =>
  SM.randn(dimensions).mul(SM.scalar(Math.sqrt(2 / dimensions[0])));

// utils

/** create a uniform dist from -k to k */
export const uniform = (dimensions: number[], k: number): SM.Tensor =>
  SM.rand(dimensions)
    .sub(SM.scalar(0.5))
    .mul(SM.scalar(k * 2));
