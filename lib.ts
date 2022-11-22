import * as SM from "@shumai/shumai";

type Layer = (input: SM.Tensor) => SM.Tensor;

/// create a sequential model
/// TODO: track previous layer size
/// TODO: track train or predict mode
export const sequential =
  (...layers: Layer[]): Layer =>
  (input) =>
    layers.reduce((input, layer) => layer(input), input);

/// linear layer (no activation fn)
export const linear = (inputSize: number, outputSize: number): Layer => {
  // following torch's `U(-sqrt(k), sqrt(k))`
  const k = Math.sqrt(1 / inputSize) * 2;
  const W = uniform([inputSize, outputSize], k).requireGrad();
  const B = uniform([outputSize], k).requireGrad();

  return (input) => input.matmul(W).add(B);
};

/// drop-out regularisation layer
/// TODO: disable when predicting
export const dropOut = (inputSize: number, keepProb: number): Layer => {
  const p = SM.scalar(keepProb);
  return (input) =>
    input
      .matmul(
        SM.rand([inputSize, inputSize]).lessThan(p).astype(SM.dtype.Float32)
      )
      .div(p);
};

/// binary cross-entropy loss
export const binaryCrossEntropy = (
  y: SM.Tensor,
  p: SM.Tensor,
  eps = 1e-6
): SM.Tensor => {
  // TODO: using clamp fn seems to break auto-grad
  const pClamp = p.maximum(SM.scalar(eps)).minimum(SM.scalar(1 - eps));
  return SM.add(
    SM.mul(y, SM.log(pClamp)),
    SM.mul(SM.scalar(1).sub(y), SM.log(SM.scalar(1).sub(pClamp)))
  );
};

/// create a uniform dist of spread k around 0
const uniform = (dimensions: number[], k: number): SM.Tensor =>
  SM.rand(dimensions).sub(SM.scalar(-0.5).mul(SM.scalar(k)));
