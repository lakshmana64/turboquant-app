/**
 * Accuracy Benchmark for TurboQuant TypeScript Port
 * 
 * Compares:
 * 1. True Inner Product
 * 2. Stage 1 only (Scalar Quantization)
 * 3. Stage 1 + Stage 2 (TurboQuant Unbiased Estimator)
 */

import * as tf from '@tensorflow/tfjs';
import { ScalarQuantizer } from '../core/scalar_quant.js';
import { UnbiasedInnerProductEstimator } from '../core/estimator.js';

async function runBenchmark() {
  const d = 128;         // Input dimension
  const m = 64;          // QJL bits (Stage 2)
  const numBits = 2;     // Bits per coordinate (Stage 1)
  const nQueries = 10;
  const nKeys = 100;

  console.log(`--- TurboQuant Benchmark (TypeScript) ---`);
  console.log(`Dim: ${d}, QJL Bits: ${m}, SQ Bits: ${numBits}`);
  console.log(`Queries: ${nQueries}, Keys: ${nKeys}\n`);

  // 1. Setup Data
  const queries = tf.randomNormal([nQueries, d]) as tf.Tensor2D;
  const keys = tf.randomNormal([nKeys, d]) as tf.Tensor2D;

  // 2. Prepare Scalar Quantizer Codebook
  await ScalarQuantizer.getCodebook(numBits);

  // 3. Stage 1: Quantize Keys
  const sqData = await ScalarQuantizer.quantize(keys, numBits);
  const keysHat = ScalarQuantizer.dequantize(sqData, numBits) as tf.Tensor2D;

  // 4. Setup TurboQuant Estimator
  const estimator = new UnbiasedInnerProductEstimator(d, m);
  
  // Encode Keys (Stage 2)
  const encodedKeys = estimator.encodeKey(keys, keysHat);
  
  // Encode Queries
  const qProjected = estimator.encodeQuery(queries) as tf.Tensor2D;

  // 5. Compute Inner Products
  
  // Ground Truth
  const trueDot = tf.matMul(queries, keys, false, true);

  // Stage 1 Only (Biased)
  const stage1Dot = tf.matMul(queries, keysHat, false, true);

  // TurboQuant (Unbiased)
  const turboDot = estimator.estimateBatch(
    queries,
    qProjected,
    keysHat,
    encodedKeys.rSigns as tf.Tensor2D,
    encodedKeys.rNorm as tf.Tensor2D
  );

  // 6. Calculate Errors (MSE)
  const trueDotArr = await trueDot.array();
  const stage1DotArr = await stage1Dot.array() as number[][];
  const turboDotArr = await turboDot.array() as number[][];

  let stage1Mse = 0;
  let turboMse = 0;
  let count = 0;

  for (let i = 0; i < nQueries; i++) {
    for (let j = 0; j < nKeys; j++) {
      const actual = (trueDotArr as number[][])[i][j];
      stage1Mse += Math.pow(actual - stage1DotArr[i][j], 2);
      turboMse += Math.pow(actual - turboDotArr[i][j], 2);
      count++;
    }
  }

  console.log(`Results:`);
  console.log(`Standard SQ MSE:   ${(stage1Mse / count).toFixed(6)}`);
  console.log(`TurboQuant MSE:    ${(turboMse / count).toFixed(6)}`);
  console.log(`Error Reduction:   ${((1 - turboMse / stage1Mse) * 100).toFixed(2)}%`);

  // Cleanup
  tf.dispose([queries, keys, sqData.indices, sqData.scales, sqData.norms, sqData.rotationMatrix, keysHat, qProjected, trueDot, stage1Dot, turboDot]);
  estimator.dispose();
}

runBenchmark().catch(console.error);
