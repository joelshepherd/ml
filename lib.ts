import * as SM from "@shumai/shumai";

type Layer = (input: SM.Tensor, train: boolean) => SM.Tensor;
type Loss = (y: SM.Tensor, p: SM.Tensor) => SM.Tensor;
type Optimiser = (
  gradients: Record<string, { grad: SM.Tensor; tensor: SM.Tensor }>
) => void;

/// create a sequential model
/// TODO: track previous layer size?
export const sequential =
  (...layers: Layer[]): Layer =>
  (input, train = false) =>
    layers.reduce((input, layer) => layer(input, train), input);

/// linear layer (no activation fn)
export const linear = (inputSize: number, outputSize: number): Layer => {
  // following torch's `U(-sqrt(k), sqrt(k))`
  const k = Math.sqrt(1 / inputSize) * 2;
  const W = uniform([inputSize, outputSize], k).requireGrad();
  const B = uniform([outputSize], k).requireGrad();
  return (input) => input.matmul(W).add(B);
};

/// relu activation
export const relu = (): Layer => {
  const t = SM.scalar(0);
  return (input) => input.maximum(t);
};

/// sigmoid activation
/// sm has native implementation
export const sigmoid = (): Layer => SM.sigmoid;

/// drop-out regularisation layer
export const dropOut = (inputSize: number, keepProb: number): Layer => {
  const p = SM.scalar(keepProb);
  return (input, train) =>
    train
      ? input
          .matmul(
            SM.rand([inputSize, inputSize]).lessThan(p).astype(SM.dtype.Float32)
          )
          .div(p)
      : input;
};

/// stochastic gradient descent optimiser
export const sgd = (learningRate: number): Optimiser => {
  const lr = SM.scalar(-learningRate);
  return (gradients) =>
    Object.values(gradients).forEach(({ tensor, grad }) => {
      if (tensor.requires_grad) {
        console.log("learn", tensor.valueOf(), "->", grad.valueOf());
        tensor.update(tensor.detach().add(tensor.grad.detach().mul(lr)));
      }
      // @ts-ignore
      tensor.grad = null;
    });
};

/// binary cross-entropy loss
export const binaryCrossEntropy =
  (eps = 1e-6): Loss =>
  (y, p) => {
    // TODO: using clamp fn seems to break auto-grad
    const pClamp = p.maximum(SM.scalar(eps)).minimum(SM.scalar(1 - eps));
    return SM.add(
      SM.mul(y, SM.log(pClamp)),
      SM.mul(SM.scalar(1).sub(y), SM.log(SM.scalar(1).sub(pClamp)))
    ).mul(SM.scalar(-1));
  };

/// train model
export const train = (
  model: Layer,
  data: Array<[y: SM.Tensor, x: SM.Tensor]>,
  opts: {
    batchSize: number;
    epochs: number;
    loss: Loss;
    optimiser: Optimiser;
  }
): void => {
  for (let epoch = 0; epoch < opts.epochs; epoch++) {
    data = SM.util.shuffle(data);

    for (let i = 0; i < data.length; i += opts.batchSize) {
      const batch = data.slice(i, i + opts.batchSize);
      // TODO: fixme
      if (batch.length < opts.batchSize) continue;
      const X = SM.concat(
        batch.map(([, x]) => x),
        -2
      );
      const Y = SM.concat(
        batch.map(([y]) => y),
        -2
      );
      const P = model(X, true);
      const loss = opts.loss(Y, P);
      console.log(loss.valueOf());
      // @ts-ignore
      opts.optimiser(loss.backward());
    }

    // TODO: metric reporter
    // TODO: validation set and early stopping
  }
};

/// create a uniform dist of spread k around 0
const uniform = (dimensions: number[], k: number): SM.Tensor =>
  SM.rand(dimensions).sub(SM.scalar(-0.5).mul(SM.scalar(k)));
