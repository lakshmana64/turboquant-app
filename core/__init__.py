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
  - turbo_formats: Turbo2/3/4 format presets (turboquant_plus)
  - polar_quant: PolarQuant algorithm (turboquant_plus)
  - sparse_v: Sparse V decoding (turboquant_plus)
  - asymmetric_kv: Asymmetric K/V support (turboquant_plus)
  - outlier: Outlier channel handling (turboquant_plus)
  - layer_adaptive: Layer-adaptive mode (turboquant_plus)
  - norm_correction: Norm correction for perplexity (turboquant_plus)
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

# TurboQuant Plus features (turboquant_plus compatibility)
from .turbo_formats import (
    TurboFormat,
    TURBO2,
    TURBO3,
    TURBO4,
    Q8_0,
    Q4_0,
    FORMAT_PRESETS,
    get_format,
    list_formats,
    create_codec_from_format,
    calculate_memory_usage
)

from .polar_quant import (
    PolarQuantConfig,
    PolarQuantCodec,
    polar_quant,
    polar_quant_roundtrip
)

from .sparse_v import (
    SparseVDecoder,
    SparseKVCache,
    apply_sparse_v_decoding
)

from .asymmetric_kv import (
    AsymmetricKVConfig,
    AsymmetricKVCache,
    create_asymmetric_cache,
    recommend_asymmetric_config
)

from .outlier import (
    OutlierConfig,
    OutlierHandler,
    OutlierAwareCodec,
    apply_outlier_aware_quantization
)

from .layer_adaptive import (
    LayerAdaptiveConfig,
    LayerAdaptiveKVCache,
    create_layer_adaptive_cache,
    recommend_layer_config
)

from .norm_correction import (
    NormCorrectionConfig,
    NormCorrector,
    NormCorrectedCodec,
    apply_norm_correction
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

    # TurboQuant Plus features (turboquant_plus compatibility)
    'TurboFormat',
    'TURBO2',
    'TURBO3',
    'TURBO4',
    'Q8_0',
    'Q4_0',
    'FORMAT_PRESETS',
    'get_format',
    'list_formats',
    'create_codec_from_format',
    'calculate_memory_usage',
    'PolarQuantConfig',
    'PolarQuantCodec',
    'polar_quant',
    'polar_quant_roundtrip',
    'SparseVDecoder',
    'SparseKVCache',
    'apply_sparse_v_decoding',
    'AsymmetricKVConfig',
    'AsymmetricKVCache',
    'create_asymmetric_cache',
    'recommend_asymmetric_config',
    'OutlierConfig',
    'OutlierHandler',
    'OutlierAwareCodec',
    'apply_outlier_aware_quantization',
    'LayerAdaptiveConfig',
    'LayerAdaptiveKVCache',
    'create_layer_adaptive_cache',
    'recommend_layer_config',
    'NormCorrectionConfig',
    'NormCorrector',
    'NormCorrectedCodec',
    'apply_norm_correction',
]
