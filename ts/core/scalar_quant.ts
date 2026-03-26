/**
 * Scalar Quantization Module for TurboQuant
 * 
 * Implements MSE-optimal Lloyd-Max scalar quantization with random rotation.
 * The rotation ensures coordinates follow a concentrated distribution (approx. Gaussian).
 */

import * as tf from '@tensorflow/tfjs';

export interface ScalarQuantizedData {
  indices: tf.Tensor;
  scales: tf.Tensor;
  norms: tf.Tensor;
  rotationMatrix: tf.Tensor2D;
}

export class ScalarQuantizer {
  private static codebookCache: Map<number, { centroids: tf.Tensor1D, boundaries: tf.Tensor1D }> = new Map();

  /**
   * Generate a random orthogonal matrix via QR decomposition.
   */
  public static generateRotationMatrix(d: number, seed: number = 42): tf.Tensor2D {
    return tf.tidy(() => {
      // Generate random Gaussian matrix
      const A = tf.randomNormal([d, d], 0, 1, 'float32', seed);
      
      // QR decomposition to get orthogonal matrix Q
      const [q, r] = tf.linalg.qr(A);
      return q as tf.Tensor2D;
    });
  }

  /**
   * Compute Lloyd-Max optimal quantization codebook for N(0, 1).
   */
  public static async getCodebook(numBits: number): Promise<{ centroids: tf.Tensor1D, boundaries: tf.Tensor1D }> {
    if (this.codebookCache.has(numBits)) {
      return this.codebookCache.get(numBits)!;
    }

    const numLevels = Math.pow(2, numBits);
    const numSamples = 100000;

    const result = tf.tidy(() => {
      // Sample from standard normal
      const samples = tf.randomNormal([numSamples], 0, 1);
      const sampleValues = Array.from(samples.dataSync());
      
      // Initialize centroids uniformly in [-4, 4]
      let centroidValues = Array.from(tf.linspace(-4, 4, numLevels).dataSync());
      
      // Iterative Lloyd-Max optimization (simplified for implementation)
      // For production, precomputing these is better.
      for (let i = 0; i < 20; i++) {
        const sums = new Array<number>(numLevels).fill(0);
        const counts = new Array<number>(numLevels).fill(0);

        for (const sample of sampleValues) {
          let bestLevel = 0;
          let bestDistance = Number.POSITIVE_INFINITY;

          for (let level = 0; level < numLevels; level++) {
            const distance = Math.abs(sample - centroidValues[level]);
            if (distance < bestDistance) {
              bestDistance = distance;
              bestLevel = level;
            }
          }

          sums[bestLevel] += sample;
          counts[bestLevel] += 1;
        }

        const newCentroidsArr: number[] = [];
        for (let l = 0; l < numLevels; l++) {
          if (counts[l] > 0) {
            newCentroidsArr.push(sums[l] / counts[l]);
          } else {
            newCentroidsArr.push(centroidValues[l]);
          }
        }
        centroidValues = newCentroidsArr;
      }
      
      const boundariesArr: number[] = [];
      for (let i = 0; i < centroidValues.length - 1; i++) {
        boundariesArr.push((centroidValues[i] + centroidValues[i + 1]) / 2);
      }
      
      return {
        centroids: tf.tensor1d(centroidValues),
        boundaries: tf.tensor1d(boundariesArr)
      };
    });

    this.codebookCache.set(numBits, result);
    return result;
  }

  /**
   * Quantize input vectors.
   */
  public static async quantize(
    x: tf.Tensor,
    numBits: number,
    rotationSeed: number = 42
  ): Promise<ScalarQuantizedData> {
    const d = x.shape[x.shape.length - 1];
    const rotationMatrix = this.generateRotationMatrix(d, rotationSeed);
    
    return tf.tidy(() => {
      const xFlat = x.reshape([-1, d]);
      
      // 1. Rotate
      const xRotated = tf.matMul(xFlat, rotationMatrix);
      
      // 2. Compute norms and scales
      const norms = tf.norm(xRotated, 2, 1, true);
      const scales = tf.div(norms, Math.sqrt(d));
      
      // 3. Normalize
      const xNormalized = tf.div(xRotated, tf.add(scales, 1e-8));
      
      // 4. Quantize using precomputed codebook
      // Note: In real TFJS, we'd use a more optimized way to digitize
      // For now, we'll do a simple comparison
      const { centroids, boundaries } = this.codebookCache.get(numBits) || { centroids: tf.tensor1d([]), boundaries: tf.tensor1d([]) };
      
      // Simple bucketization
      const indices = tf.sum(tf.cast(tf.greater(xNormalized.expandDims(-1), boundaries.expandDims(0).expandDims(0)), 'int32'), -1);
      
      return {
        indices: indices.reshape(x.shape),
        scales: scales.reshape([...x.shape.slice(0, -1), 1]),
        norms: norms.reshape([...x.shape.slice(0, -1), 1]),
        rotationMatrix
      };
    });
  }

  /**
   * Dequantize indices back to vectors.
   */
  public static dequantize(
    data: ScalarQuantizedData,
    numBits: number
  ): tf.Tensor {
    const { centroids } = this.codebookCache.get(numBits)!;
    
    return tf.tidy(() => {
      const d = data.rotationMatrix.shape[0];
      const indicesFlat = data.indices.reshape([-1, d]);
      const scalesFlat = data.scales.reshape([-1, 1]);
      
      // 1. Lookup centroids
      const xNormalized = tf.gather(centroids, indicesFlat);
      
      // 2. Rescale
      const xScaled = tf.mul(xNormalized, scalesFlat);
      
      // 3. Inverse rotation (Transpose of orthogonal matrix)
      const xRotated = tf.matMul(xScaled, data.rotationMatrix, false, true);
      
      return xRotated.reshape(data.indices.shape);
    });
  }
}
