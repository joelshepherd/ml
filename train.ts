import * as SM from "@shumai/shumai";
import * as Data from "./data.js";
import * as Model from "./model.js";
import * as Metric from "./metric.js";

/** example */
export type Example = [feature: SM.Tensor, label: SM.Tensor];

// loss fns

/** loss fn */
type Loss = (y: SM.Tensor, p: SM.Tensor) => SM.Tensor;

/** binary cross-entropy loss */
export const binaryCrossEntropy = (): Loss => (y, p) =>
  SM.mul(
    SM.scalar(-1),
    SM.mean(
      SM.add(
        SM.mul(y, clippedLog(p)),
        SM.mul(SM.scalar(1).sub(y), clippedLog(SM.scalar(1).sub(p)))
      )
    )
  );

/**
 * binary cross-entrioy loss from logits
 * improved numerical stability vs. sigmoid & bce separately
 */
export const binaryCrossEntropyFromLogits = (): Loss => (y, p) =>
  SM.mean(
    SM.add(
      SM.sub(SM.maximum(p, SM.scalar(0)), SM.mul(p, y)),
      SM.log1p(SM.exp(SM.negate(SM.abs(p))))
    )
  );

/** cross-entropy loss */
// TODO: if batch size is 1, throws
export const crossEntropy = (): Loss => (y, p) =>
  SM.mul(SM.scalar(-1), SM.mean(SM.sum(SM.mul(y, clippedLog(p)), [1])));

/** mean absolute error loss */
export const meanAbsoluteError = (): Loss => (y, p) =>
  SM.mean(SM.abs(SM.sub(y, p)));

/** mean squared error loss */
export const meanSquaredError = (): Loss => (y, p) => {
  const diff = SM.sub(y, p);
  return SM.mean(SM.mul(diff, diff));
};

// training

/**
 * fit model
 * TODO: early stopping
 */
export const fit = (
  model: Model.Layer,
  train: Data.Dataset<Example>,
  validation: Data.Dataset<Example>,
  opts: {
    epochs?: number;
    loss: Loss;
    metrics?: BatchMetric[];
    optimiser: Optimiser;
  }
): void => {
  // defaults
  opts.epochs ??= 1;
  opts.metrics ??= [];

  // metrics
  const lossMetric = batchMetric("loss", opts.loss, SM.mean);
  const restMetrics = opts.metrics;
  const metrics = [lossMetric, ...restMetrics];

  // training loop
  for (let epoch = 1; epoch <= opts.epochs; epoch++) {
    console.log(bold(`epoch ${epoch}/${opts.epochs}`));

    // training
    for (const [feature, label] of train()) {
      const prediction = model(feature, true);
      const loss = lossMetric.add(label, prediction);
      collectBatchMetrics(restMetrics, label, prediction);
      // @ts-ignore not over network, will not be a promise
      opts.optimiser(loss.backward());
    }
    printBatchMetrics("training", metrics);

    // validation
    for (const [feature, label] of validation())
      collectBatchMetrics(metrics, label, model(feature));
    printBatchMetrics("validation", metrics);

    console.log();
  }
};

// FIXME: side-effects
const collectBatchMetrics = (
  metrics: BatchMetric[],
  label: SM.Tensor,
  prediction: SM.Tensor
): void => metrics.forEach((metric) => metric.add(label, prediction));

// FIXME: side-effects
const printBatchMetrics = (phase: string, metrics: BatchMetric[]): void => {
  console.log(underline(phase));
  metrics.forEach((metric) => {
    const output = metric.reduce();
    console.log(`${metric.name}:`, output ? printTensor(output) : null);
  });
};

// optimisers

/** optimiser */
type Optimiser = (
  gradients: Record<string, { grad: SM.Tensor; tensor: SM.Tensor }>
) => void;

/** stochastic gradient descent optimiser */
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

// fit metrics

/** batch metric for training */
// TODO: FitMetric?
// TODO: potentially add a print function?
type BatchMetric = {
  name: string;
  add: Metric.Metric;
  reduce: () => SM.Tensor | null;
};

/** wrap metric for batching */
const batchMetric = (
  name: string,
  metric: Metric.Metric,
  reducer: (state: SM.Tensor, axes: number[]) => SM.Tensor
): BatchMetric => {
  let state: SM.Tensor | null = null;
  return {
    name,
    add: (y, p) => {
      const output = metric(y, p);
      state = state
        ? SM.concat([state, output], 0)
        : SM.reshape(output, [1].concat(output.shape));
      return output;
    },
    reduce: () => {
      let output = null;
      if (state) {
        output = reducer(state, [0]);
        state = null;
      }
      return output;
    },
  };
};

/** accuracy fit metric */
export const accuracy = batchMetric("accuracy", Metric.accuracy, SM.mean);

/** confusion fit metric */
export const confusion = batchMetric("confusion", Metric.confusion, SM.sum);

// utils

const bold = (text: string): string => `\033[1m${text}\033[0m`;

const underline = (text: string): string => `\033[4m${text}\033[0m`;

const clippedLog = (tensor: SM.Tensor): SM.Tensor =>
  SM.maximum(SM.log(tensor), SM.scalar(-100));

type PrintedTensor = number | Array<PrintedTensor>;

const printTensor = (tensor: SM.Tensor): PrintedTensor => {
  // TODO: show brackets for a dimension with only 1 value
  if (tensor.elements === 1) return tensor.valueOf();
  return new Array(tensor.shape[0])
    .fill(null)
    .map((_, i) =>
      printTensor(
        tensor.index(tensor.shape.map((_, ii) => (ii === 0 ? i : ":")))
      )
    );
};
