/**
 * TurboQuant SDK - High-level API for Unbiased Quantization
 * 
 * Provides a simple, unified interface for the two-stage TurboQuant
 * quantization scheme in TypeScript.
 */

import * as tf from '@tensorflow/tfjs';
import { ScalarQuantizer } from '../core/scalar_quant.js';
import { UnbiasedInnerProductEstimator, EncodedKey } from '../core/estimator.js';

export interface TurboEncodedData {
  indices: tf.Tensor;
  scales: tf.Tensor;
  rSigns: tf.Tensor;
  rNorm: tf.Tensor;
  xHat: tf.Tensor;
}

export class TurboQuantizer {
  public readonly inputDim: number;
  public readonly qjlBits: number;
  public readonly sqBits: number;
  private readonly estimator: UnbiasedInnerProductEstimator;
  private rotationMatrix: tf.Tensor2D;

  constructor(
    inputDim: number,
    qjlBits: number = 64,
    sqBits: number = 2,
    seed: number = 42
  ) {
    this.inputDim = inputDim;
    this.qjlBits = qjlBits;
    this.sqBits = sqBits;
    
    // Initialize components
    this.rotationMatrix = ScalarQuantizer.generateRotationMatrix(inputDim, seed);
    this.estimator = new UnbiasedInnerProductEstimator(inputDim, qjlBits, seed);
  }

  /**
   * Initialize codebooks (required before first encode/decode).
   */
  public async init() {
    await ScalarQuantizer.getCodebook(this.sqBits);
  }

  /**
   * Encode (quantize) a tensor using the two-stage scheme.
   */
  public async encode(x: tf.Tensor): Promise<TurboEncodedData> {
    // 1. Stage 1: Scalar Quantization
    const sqData = await ScalarQuantizer.quantize(x, this.sqBits);
    
    // Reconstruction for residual
    const xHat = ScalarQuantizer.dequantize(sqData, this.sqBits);
    
    // 2. Stage 2: QJL Residual Encoding
    const encodedKey = this.estimator.encodeKey(x, xHat);
    
    return {
      indices: sqData.indices,
      scales: sqData.scales,
      rSigns: encodedKey.rSigns,
      rNorm: encodedKey.rNorm,
      xHat: xHat
    };
  }

  /**
   * Estimate inner product <q, x>
   */
  public estimate(q: tf.Tensor, encoded: TurboEncodedData): tf.Tensor {
    const qProjected = this.estimator.encodeQuery(q);
    
    const key: EncodedKey = {
      xHat: encoded.xHat,
      rSigns: encoded.rSigns,
      rNorm: encoded.rNorm
    };
    
    return this.estimator.estimate(q, qProjected, key);
  }

  /**
   * Batch estimate inner products (queries x keys)
   */
  public estimateBatch(queries: tf.Tensor2D, encodedKeys: TurboEncodedData): tf.Tensor2D {
    const qProjected = this.estimator.encodeQuery(queries) as tf.Tensor2D;
    
    return this.estimator.estimateBatch(
      queries,
      qProjected,
      encodedKeys.xHat as tf.Tensor2D,
      encodedKeys.rSigns as tf.Tensor2D,
      encodedKeys.rNorm as tf.Tensor2D
    );
  }

  /**
   * Free memory
   */
  public dispose() {
    this.rotationMatrix.dispose();
    this.estimator.dispose();
  }
}

/**
 * One-line API to quantize a tensor and return the quantizer instance.
 */
export async function optimize(
  x: tf.Tensor,
  qjlBits: number = 64,
  sqBits: number = 2,
  seed: number = 42
): Promise<[TurboEncodedData, TurboQuantizer]> {
  const d = x.shape[x.shape.length - 1];
  const quantizer = new TurboQuantizer(d, qjlBits, sqBits, seed);
  await quantizer.init();
  const encoded = await quantizer.encode(x);
  return [encoded, quantizer];
}
