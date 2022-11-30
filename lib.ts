import * as SM from "@shumai/shumai";

type Layer = (input: SM.Tensor, train?: boolean) => SM.Tensor;
type Loss = (y: SM.Tensor, p: SM.Tensor) => SM.Tensor;
type Optimiser = (
  gradients: Record<string, { grad: SM.Tensor; tensor: SM.Tensor }>
) => void;
type Metric = (y: SM.Tensor, p: SM.Tensor) => number;
type DataSet = [train: () => Generator<Batch>, validation: Batch];
type Batch = [y: SM.Tensor, x: SM.Tensor];

/// create a sequential model
/// TODO: track previous layer size?
export const sequential =
  (...layers: Layer[]): Layer =>
  (input, train = false) =>
    layers.reduce((input, layer) => layer(input, train), input);

/// linear layer (no activation fn)
export const linear = (inputSize: number, outputSize: number): Layer => {
  // following torch's `U(-sqrt(k), sqrt(k))`
  const k = Math.sqrt(1 / inputSize);
  const W = uniform([inputSize, outputSize], k).requireGrad();
  // TODO: biases seems to adjust the number of inputs when optimising
  // const B = uniform([outputSize], k).requireGrad();
  return (input) => input.matmul(W); // .add(B);
};

/// relu activation
export const relu = (): Layer => {
  const t = SM.scalar(0);
  return (input) => input.maximum(t);
};

/// sigmoid activation
/// sm has native implementation
export const sigmoid = (): Layer => SM.sigmoid;

/// softmax activation
export const softmax = (): Layer => (input) => {
  const e = input.exp();
  return e.div(e.sum([-1], true));
};

/// drop-out regularisation layer
export const dropOut = (keepProb: number): Layer => {
  const p = SM.scalar(keepProb);
  return (input, train) =>
    train ? input.mul(SM.rand(input.shape).lessThan(p)).div(p) : input;
};

/// stochastic gradient descent optimiser
export const sgd = (learningRate: number): Optimiser => {
  const lr = SM.scalar(-learningRate);
  return (gradients) =>
    Object.values(gradients).forEach(({ tensor, grad }) => {
      if (tensor.requires_grad) {
        tensor.update(tensor.detach().add(grad.detach().mul(lr)));
      }
      // @ts-ignore
      tensor.grad = null;
    });
};

/// binary cross-entropy loss
export const binaryCrossEntropy = (eps = 1e-6): Loss => {
  const low = SM.scalar(eps);
  const high = SM.scalar(1 - eps);
  return (y, p) => {
    // TODO: y seems to come in dim [n], while p comes in dim [n,1]
    //       this seems to expand the output to [n,n] if i do not reshape
    //       -> i've moved this the train fn for now
    // TODO: clamp seems to break auto-grad
    // p = SM.clamp(p, eps, 1 - eps);
    p = p.maximum(low).minimum(high);
    return SM.mul(
      SM.scalar(-1),
      SM.add(
        SM.mul(y, SM.log(p)),
        SM.mul(SM.scalar(1).sub(y), SM.log(SM.scalar(1).sub(p)))
      )
    ).mean();
  };
};

/// cross-entropy loss
// TODO: not currently numerically stable
export const crossEntropy = (eps = 1e-6): Loss => {
  const low = SM.scalar(eps);
  const high = SM.scalar(1 - eps);
  return (y, p) => {
    p = p.maximum(low).minimum(high);
    return SM.mul(
      SM.scalar(-1),
      SM.sum(SM.mul(y, SM.log(p)), [-1], true)
    ).mean();
  };
};

/// data set
// TODO: data that cannot fit into memory?
// TODO: shuffle - need better `.index()` options first?
export const dataSet = (
  data: SM.Tensor,
  opts: {
    xIndex: number | string;
    yIndex: number | string;
    batchSize: number;
    validationPortion: number;
  }
): DataSet => {
  const [len] = data.shape;
  const split = Math.floor(len * (1 - opts.validationPortion));
  const batch = (rowIndex: string): Batch => [
    data.index([rowIndex, opts.yIndex]),
    data.index([rowIndex, opts.xIndex]),
  ];
  return [
    function* () {
      // TODO: batching validation when metrics are ready
      // const min = train ? 0 : split;
      // const max = train ? split : len;
      for (let i = 0; i < split; i += opts.batchSize)
        yield batch(i + ":" + Math.min(i + opts.batchSize, split));
    },
    batch(split + ":" + len),
  ];
};

/// train model
// TODO: handle p reshaping
export const train = (
  model: Layer,
  [train, validation]: DataSet,
  opts: {
    epochs: number;
    loss: Loss;
    metrics: Record<string, Metric>;
    optimiser: Optimiser;
  }
): void => {
  const metrics: Record<string, Metric> = {
    loss: (y, p) => opts.loss(y, p).valueOf(),
    ...opts.metrics,
  };

  for (let epoch = 0; epoch < opts.epochs; epoch++) {
    console.log(bold(`Epoch ${epoch + 1}/${opts.epochs}`));

    for (const [y, x] of train()) {
      const p = model(x, true).reshape(y.shape);
      const loss = opts.loss(y, p);
      // @ts-ignore not over network, will not be a promise
      opts.optimiser(loss.backward());
    }

    // metrics
    const [y, x] = validation;
    const p = model(x).reshape(y.shape);
    Object.entries(metrics).forEach(([name, metric]) =>
      console.log(`${name}:`.padEnd(10), metric(y, p))
    );
    console.log();

    // TODO: early stopping
  }
};

// TODO: metrics for multiple classes
export const accuracy = (): Metric => (y, p) =>
  (y.elements -
    (y.shape[1]
      ? y.argmax(1).sub(p.argmax(1))
      : y.sub(p.greaterThan(SM.scalar(0.5)))
    )
      .countNonzero()
      .valueOf()) /
  y.elements;

export const precision = (): Metric => (y, p) => {
  const m = [
    [0, 0],
    [0, 0],
  ];
  const len = y.shape[0];
  for (let i = 0; i < len; i++) {
    const i1 = y.index([i]).valueOf();
    const i2 = p.index([i]).valueOf() > 0.5 ? 1 : 0;
    m[i1][i2] += 1;
  }
  return m[1][1] / (m[1][1] + m[0][1] || 1);
};

export const f1 = (): Metric => (y, p) => {
  const m = [
    [0, 0],
    [0, 0],
  ];
  const len = y.shape[0];
  for (let i = 0; i < len; i++) {
    const i1 = y.index([i]).valueOf();
    const i2 = p.index([i]).valueOf() > 0.5 ? 1 : 0;
    m[i1][i2] += 1;
  }
  return (2 * m[1][1]) / (2 * m[1][1] + m[0][1] + m[1][0]);
};

/// create a uniform dist from -k to k
export const uniform = (dimensions: number[], k: number): SM.Tensor =>
  SM.rand(dimensions)
    .sub(SM.scalar(0.5))
    .mul(SM.scalar(k * 2));

// TODO: \" in quotes
export const csvLine = (line: string): string[] => {
  const cells = [];
  let cell = "";
  let inQuotes = false;
  for (let letter of line) {
    if (letter === '"') {
      inQuotes = !inQuotes;
    } else if (!inQuotes && letter === ",") {
      cells.push(cell);
      cell = "";
    } else {
      cell += letter;
    }
  }
  cells.push(cell);
  return cells;
};

export const tableToTensor = (rows: number[][]) =>
  SM.tensor(Float32Array.from(rows.flat())).reshape([
    rows.length,
    rows[0].length,
  ]);

const bold = (text: string): string => `\033[1m${text}\033[0m`;
