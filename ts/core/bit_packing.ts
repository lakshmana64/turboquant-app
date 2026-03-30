/**
 * Bit-packing utilities for TurboQuant (TypeScript).
 * 
 * Allows packing low-bit indices (1, 2, 4 bits) into Uint8Arrays
 * for memory-efficient storage in browser/Node environments.
 */

import * as tf from '@tensorflow/tfjs';

export class BitPacker {
  /**
   * Pack integer tensor into Uint8Array.
   */
  public static async pack(x: tf.Tensor, bits: number): Promise<Uint8Array> {
    if (![1, 2, 4, 8].includes(bits)) {
      throw new Error(`Unsupported bit width: ${bits}`);
    }

    const data = await x.data();
    const elementsPerByte = 8 / bits;
    const packedSize = Math.ceil(data.length / elementsPerByte);
    const packed = new Uint8Array(packedSize);

    if (bits === 8) {
      return new Uint8Array(data);
    }

    let byteIdx = 0;
    let bitShift = 0;

    for (let i = 0; i < data.length; i++) {
      packed[byteIdx] |= (data[i] & ((1 << bits) - 1)) << bitShift;
      bitShift += bits;

      if (bitShift >= 8) {
        bitShift = 0;
        byteIdx++;
      }
    }

    return packed;
  }

  /**
   * Unpack Uint8Array back into an integer tensor.
   */
  public static unpack(packed: Uint8Array, bits: number, shape: number[]): tf.Tensor {
    if (bits === 8) {
      return tf.tensor(Array.from(packed), shape, 'int32');
    }

    const numElements = shape.reduce((a, b) => a * b, 1);
    const unpacked = new Int32Array(numElements);
    const elementsPerByte = 8 / bits;
    const mask = (1 << bits) - 1;

    for (let i = 0; i < numElements; i++) {
      const byteIdx = Math.floor(i / elementsPerByte);
      const bitShift = (i % elementsPerByte) * bits;
      unpacked[i] = (packed[byteIdx] >> bitShift) & mask;
    }

    return tf.tensor(unpacked, shape, 'int32');
  }
}
