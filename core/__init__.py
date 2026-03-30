"""
TurboQuant Core - Quantization & Estimation Engine

Modules:
  - scalar_quant: MSE-optimal scalar quantization
  - qjl_projection: Quantized Johnson-Lindenstrauss projection
  - residual: Residual computation and encoding
  - estimator: Unbiased inner product estimator
  - codec: Two-stage TurboQuant codec
  - optimized: GPU-accelerated implementations
  - streaming: Memory-efficient streaming encoder
  - mixed_precision: FP8/INT8 quantization support
  - monitoring: Metrics and logging
  - aoti: AOTInductor compilation and export
  - distributed: Multi-GPU support
  - bit_packing: Bit-packing utilities for low-bit indices
"""

from .bit_packing import (
    pack_bits,
    unpack_bits,
    pack_signs,
    unpack_signs
)

from .scalar_quant import (
    quantize_scalar,
    dequantize_scalar,
    get_codebook,
    quantize_and_reconstruct
)

from .qjl_projection import (
    QJLProjection,
    create_qjl_projection
)

from .residual import (
    compute_residual,
    compute_residual_norm,
    encode_residual_qjl,
    decode_residual_correction,
    ResidualEncoder
)

from .estimator import (
    UnbiasedInnerProductEstimator,
    estimate_inner_product_unbiased,
    estimate_inner_product_batch,
    validate_unbiasedness
)

from .codec import (
    TurboQuantCodec,
    TurboQuantConfig,
    EncodedKey,
    create_codec
)

from .optimized import (
    TurboQuantCodecOptimized,
    QJLProjectionOptimized,
    create_optimized_codec
)

from .streaming import (
    StreamingEncoder,
    KVCacheStreamer,
    stream_encode
)

from .mixed_precision import (
    MixedPrecisionQuantizer,
    MixedPrecisionCodec,
    LowPrecisionAttention,
    FP8_AVAILABLE
)

from .monitoring import (
    MetricsCollector,
    TurboQuantLogger,
    OperationMetrics,
    SessionStats,
    get_logger,
    enable_logging,
    track_operation
)

from .aoti import (
    CompiledTurboQuantCodec,
    compile_codec,
    export_aot_inductor,
    benchmark_compiled
)

from .distributed import (
    DistributedStreamingEncoder,
    DistributedKVCacheStreamer
)

__all__ = [
    # Scalar quantization
    'quantize_scalar',
    'dequantize_scalar',
    'get_codebook',
    'quantize_and_reconstruct',
    
    # QJL projection
    'QJLProjection',
    'create_qjl_projection',
    
    # Residual
    'compute_residual',
    'compute_residual_norm',
    'encode_residual_qjl',
    'decode_residual_correction',
    'ResidualEncoder',
    
    # Estimator
    'UnbiasedInnerProductEstimator',
    'estimate_inner_product_unbiased',
    'estimate_inner_product_batch',
    'validate_unbiasedness',
    
    # Codec
    'TurboQuantCodec',
    'TurboQuantConfig',
    'EncodedKey',
    'create_codec',
    
    # Optimized (Phase 1)
    'TurboQuantCodecOptimized',
    'QJLProjectionOptimized',
    'create_optimized_codec',
    
    # Streaming (Phase 1)
    'StreamingEncoder',
    'KVCacheStreamer',
    'stream_encode',
    
    # Mixed Precision (Phase 2)
    'MixedPrecisionQuantizer',
    'MixedPrecisionCodec',
    'LowPrecisionAttention',
    'FP8_AVAILABLE',
    
    # Monitoring (Phase 3)
    'MetricsCollector',
    'TurboQuantLogger',
    'OperationMetrics',
    'SessionStats',
    'get_logger',
    'enable_logging',
    'track_operation',

    # AOTI (Phase 3)
    'CompiledTurboQuantCodec',
    'compile_codec',
    'export_aot_inductor',
    'benchmark_compiled',
    
    # Distributed (Phase 3)
    'DistributedStreamingEncoder',
    'DistributedKVCacheStreamer',
    
    # Bit-packing
    'pack_bits',
    'unpack_bits',
    'pack_signs',
    'unpack_signs',
]
