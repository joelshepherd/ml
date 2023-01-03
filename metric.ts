import * as SM from "@shumai/shumai";

/** metric */
export type Metric = (y: SM.Tensor, p: SM.Tensor) => SM.Tensor;

/** accuracy metric */
// TODO(perf): how much performance could be gained by splitting
//             binary and multi-class increase performance
//             to remove the extra if condition
export const accuracy: Metric = (y, p) =>
  (y.shape[1] > 1
    ? // multi-class
      y.argmax(-1, true).sub(p.argmax(-1, true)).sum([-1])
    : // binary
      y.sub(p.greaterThan(SM.scalar(0.5)))
  )
    .eq(SM.scalar(0))
    .mean();

/** confusion matrix metric */
// TODO: sum reduction type
export const confusion: Metric = (y, p) =>
  // TODO: binary case - `where(gt(0.5), [0,1], [1,0])`?
  // TODO: fix case for 2 max classes, argmax + onehot would work, if we had onehot!
  // convert probabilities to just top prediction
  SM.matmul(y.T(), SM.eq(p, SM.amax(p, [-1], true)).astype(SM.dtype.Float32));

/**
 * precision metric
 * TODO: multi-class
 */
// export const precision = (): Metric => (y, p) => {
//   const m = [
//     [0, 0],
//     [0, 0],
//   ];
//   const len = y.shape[0];
//   for (let i = 0; i < len; i++) {
//     const i1 = y.index([i, 0]).valueOf();
//     const i2 = p.index([i, 0]).valueOf() > 0.5 ? 1 : 0;
//     m[i1][i2] += 1;
//   }
//   return m[1][1] / (m[1][1] + m[0][1] || 1);
// };

// TODO: recall

/**
 * f1 metric
 * TODO: multi-class
 */
// export const f1 = (): Metric => (y, p) => {
//   const m = [
//     [0, 0],
//     [0, 0],
//   ];
//   const len = y.shape[0];
//   for (let i = 0; i < len; i++) {
//     const i1 = y.index([i, 0]).valueOf();
//     const i2 = p.index([i, 0]).valueOf() > 0.5 ? 1 : 0;
//     m[i1][i2] += 1;
//   }
//   return (2 * m[1][1]) / (2 * m[1][1] + m[0][1] + m[1][0]);
// };
