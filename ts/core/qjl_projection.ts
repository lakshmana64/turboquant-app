/**
 * Quantized Johnson-Lindenstrauss (QJL) Projection for TurboQuant
 * 
 * Implements the 1-bit QJL transform for residual encoding.
 * Captures residual information lost during scalar quantization.
 */

import * as tf from '@tensorflow/tfjs';

export class QJLProjection {
  public readonly inputDim: number;
  public readonly outputDim: number;
  private readonly seed: number;
  private projectionMatrix: tf.Tensor2D;

  constructor(inputDim: number, outputDim: number, seed: number = 42) {
    this.inputDim = inputDim;
    this.outputDim = outputDim;
    this.seed = seed;
    
    // Generate projection matrix R (m x d)
    this.projectionMatrix = this.generateProjectionMatrix();
  }

  /**
   * Generate a random Gaussian projection matrix with fixed seed.
   */
  private generateProjectionMatrix(): tf.Tensor2D {
    // Generate matrix R with entries ~ N(0, 1)
    const R = tf.randomNormal(
      [this.outputDim, this.inputDim],
      0,
      1,
      'float32',
      this.seed
    );

    // NOTE: We do NOT normalize rows as the scaling factor assumes N(0, 1)
    
    return R as tf.Tensor2D;
  }

  /**
   * Apply QJL projection to input vectors (Query projection).
   * Computes R @ x.
   */
  public project(x: tf.Tensor): tf.Tensor {
    // x: (..., d)
    const origShape = x.shape;
    const batchDims = origShape.slice(0, -1);
    const xFlat = x.reshape([-1, this.inputDim]);

    // Project: (N, m) = (N, d) @ (m, d).T
    const projected = tf.matMul(xFlat, this.projectionMatrix, false, true);
    
    const finalShape = [...batchDims, this.outputDim];
    return projected.reshape(finalShape);
  }

  /**
   * Apply QJL projection and 1-bit quantization (Sign encoding).
   * This is used for residual encoding.
   */
  public projectAndQuantize(r: tf.Tensor): tf.Tensor {
    const projected = this.project(r);
    
    // 1-bit quantization: sign(x)
    // map 0 to 1 for consistency
    const signs = tf.sign(projected);
    const nonZeroSigns = tf.where(
        tf.equal(signs, 0),
        tf.onesLike(signs),
        signs
    );
    
    return nonZeroSigns;
  }

  /**
   * Estimate inner product <q, r> from QJL encoding.
   * Formula: <q, r> ≈ sqrt(π/2) * ||r|| / m * <R@q, sign(R@r)>
   */
  public estimateInnerProduct(
    qProjected: tf.Tensor,
    rSigns: tf.Tensor,
    residualNorm: tf.Tensor
  ): tf.Tensor {
    const m = this.outputDim;
    
    // Scaling factor: sqrt(π/2) / m
    const scale = Math.sqrt(Math.PI / 2) / m;
    
    // Inner product in projected space: <Rq, sign(Rr)>
    const dotProduct = tf.sum(tf.mul(qProjected, rSigns), -1, true);
    
    // Final estimate: scale * ||r|| * <Rq, sign(Rr)>
    const estimate = tf.mul(tf.mul(scale, residualNorm), dotProduct);
    
    return estimate.squeeze([-1]);
  }

  /**
   * Free memory (TFJS specific)
   */
  public dispose() {
    this.projectionMatrix.dispose();
  }
}
