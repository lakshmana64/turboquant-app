/**
 * Unbiased Inner Product Estimator for TurboQuant
 * 
 * Implements the two-stage quantization scheme in TypeScript using TensorFlow.js.
 */

import * as tf from '@tensorflow/tfjs';
import { QJLProjection } from './qjl_projection.js';

export interface EncodedKey {
  xHat: tf.Tensor;
  rSigns: tf.Tensor;
  rNorm: tf.Tensor;
}

export class UnbiasedInnerProductEstimator {
  public readonly inputDim: number;
  public readonly outputDim: number;
  private readonly qjl: QJLProjection;

  constructor(inputDim: number, outputDim: number, seed: number = 42) {
    this.inputDim = inputDim;
    this.outputDim = outputDim;
    this.qjl = new QJLProjection(inputDim, outputDim, seed);
  }

  /**
   * Encode a key vector for inner product estimation.
   * Expects x (original) and x_hat (reconstructed from Stage 1).
   */
  public encodeKey(x: tf.Tensor, xHat: tf.Tensor): EncodedKey {
    // x, xHat: (..., d)
    const residual = tf.sub(x, xHat);
    
    // Compute residual norm before quantization
    const rNorm = tf.norm(residual, 2, -1, true);
    
    // Stage 2: QJL encoding of residual
    const rSigns = this.qjl.projectAndQuantize(residual);
    
    return {
      xHat,
      rSigns,
      rNorm
    };
  }

  /**
   * Encode a query vector (just projection).
   */
  public encodeQuery(q: tf.Tensor): tf.Tensor {
    return this.qjl.project(q);
  }

  /**
   * Estimate inner product <q, x> using reconstruction + correction.
   * <q, x> ≈ <q, x_hat> + correction
   */
  public estimate(
    q: tf.Tensor,
    qProjected: tf.Tensor,
    key: EncodedKey
  ): tf.Tensor {
    // Stage 1 contribution: <q, x_hat>
    const baseDot = tf.sum(tf.mul(q, key.xHat), -1);
    
    // Stage 2 contribution: correction from QJL
    const correction = this.qjl.estimateInnerProduct(
      qProjected,
      key.rSigns,
      key.rNorm
    );
    
    return tf.add(baseDot, correction);
  }

  /**
   * Batch estimate inner products for multiple queries and keys.
   * (n_q, d) x (n_k, d) -> (n_q, n_k)
   */
  public estimateBatch(
    queries: tf.Tensor2D,
    qProjected: tf.Tensor2D,
    keysHat: tf.Tensor2D,
    rSigns: tf.Tensor2D,
    rNorms: tf.Tensor2D
  ): tf.Tensor2D {
    // Stage 1: (n_q, d) @ (n_k, d).T
    const baseDots = tf.matMul(queries, keysHat, false, true);
    
    // Stage 2: (n_q, m) @ (n_k, m).T
    const projectedDots = tf.matMul(qProjected, rSigns, false, true);
    
    // Scale: sqrt(pi/2) / m
    const scale = Math.sqrt(Math.PI / 2) / this.outputDim;
    
    // Scaled correction: scale * <Rq_i, sign_j> * ||r_j||
    // rNorms: (n_k, 1) -> (1, n_k)
    const correction = tf.mul(
      tf.mul(scale, projectedDots),
      rNorms.transpose()
    );
    
    return tf.add(baseDots, correction) as tf.Tensor2D;
  }

  /**
   * Free memory
   */
  public dispose() {
    this.qjl.dispose();
  }
}
