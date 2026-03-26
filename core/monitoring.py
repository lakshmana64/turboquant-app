"""
TurboQuant Monitoring & Metrics

Logging, metrics, and monitoring for TurboQuant operations.

Features:
  - Operation timing
  - Memory tracking
  - Quality metrics
  - Compression statistics
  - Export to Prometheus/Grafana

Usage:
    from turboquant.monitoring import MetricsCollector, TurboQuantLogger
    
    # Enable logging
    logger = TurboQuantLogger(level="INFO")
    
    # Collect metrics
    collector = MetricsCollector()
    
    with collector.track_operation("encode"):
        encoded = codec.encode_keys_batch(keys)
    
    # Get stats
    stats = collector.get_stats()
    print(f"Avg encode time: {stats['encode']['avg_time_ms']:.2f}ms")
"""

import time
import logging
import json
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field, asdict
from contextlib import contextmanager
from datetime import datetime
import threading


@dataclass
class OperationMetrics:
    """Metrics for a single operation."""
    
    name: str
    count: int = 0
    total_time_ms: float = 0
    min_time_ms: float = float('inf')
    max_time_ms: float = 0
    memory_bytes: int = 0
    compression_ratio: float = 1.0
    quality_score: float = 1.0
    
    @property
    def avg_time_ms(self) -> float:
        return self.total_time_ms / max(self.count, 1)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'count': self.count,
            'avg_time_ms': self.avg_time_ms,
            'min_time_ms': self.min_time_ms if self.min_time_ms != float('inf') else 0,
            'max_time_ms': self.max_time_ms,
            'memory_bytes': self.memory_bytes,
            'compression_ratio': self.compression_ratio,
            'quality_score': self.quality_score,
        }


@dataclass
class SessionStats:
    """Statistics for a session."""
    
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    total_operations: int = 0
    total_encode_time_ms: float = 0
    total_decode_time_ms: float = 0
    total_query_time_ms: float = 0
    total_memory_saved_mb: float = 0
    avg_compression_ratio: float = 1.0
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MetricsCollector:
    """
    Collects and aggregates metrics for TurboQuant operations.
    
    Thread-safe for concurrent operations.
    """
    
    def __init__(self, window_size: int = 1000):
        """
        Initialize metrics collector.
        
        Args:
            window_size: Number of recent operations to track
        """
        self.window_size = window_size
        self._lock = threading.Lock()
        self._metrics: Dict[str, OperationMetrics] = {}
        self._recent_times: Dict[str, List[float]] = {}
        self._session = SessionStats()
    
    @contextmanager
    def track_operation(self, name: str):
        """
        Context manager for tracking operation timing.
        
        Usage:
            with collector.track_operation("encode"):
                encoded = codec.encode_keys_batch(keys)
        """
        start_time = time.perf_counter()
        try:
            yield
        except Exception as e:
            with self._lock:
                self._session.errors.append(f"{name}: {str(e)}")
            raise
        finally:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self._record_metric(name, elapsed_ms)
    
    def _record_metric(self, name: str, elapsed_ms: float):
        """Record a metric."""
        with self._lock:
            # Update operation metrics
            if name not in self._metrics:
                self._metrics[name] = OperationMetrics(name=name)
            
            metrics = self._metrics[name]
            metrics.count += 1
            metrics.total_time_ms += elapsed_ms
            metrics.min_time_ms = min(metrics.min_time_ms, elapsed_ms)
            metrics.max_time_ms = max(metrics.max_time_ms, elapsed_ms)
            
            # Update recent times
            if name not in self._recent_times:
                self._recent_times[name] = []
            self._recent_times[name].append(elapsed_ms)
            
            # Keep window size
            if len(self._recent_times[name]) > self.window_size:
                self._recent_times[name].pop(0)
            
            # Update session stats
            self._session.total_operations += 1
            
            if name == 'encode':
                self._session.total_encode_time_ms += elapsed_ms
            elif name == 'decode':
                self._session.total_decode_time_ms += elapsed_ms
            elif name == 'query':
                self._session.total_query_time_ms += elapsed_ms
    
    def record_memory(self, operation: str, bytes_used: int):
        """Record memory usage for an operation."""
        with self._lock:
            if operation in self._metrics:
                self._metrics[operation].memory_bytes = bytes_used
    
    def record_compression(
        self,
        operation: str,
        ratio: float,
        original_bytes: int,
        compressed_bytes: int
    ):
        """Record compression statistics."""
        with self._lock:
            if operation in self._metrics:
                metrics = self._metrics[operation]
                metrics.compression_ratio = ratio
                self._session.total_memory_saved_mb += (
                    (original_bytes - compressed_bytes) / 1e6
                )
    
    def record_quality(self, operation: str, score: float):
        """Record quality metric (e.g., correlation, MSE)."""
        with self._lock:
            if operation in self._metrics:
                self._metrics[operation].quality_score = score
    
    def get_stats(self) -> Dict[str, Any]:
        """Get all statistics."""
        with self._lock:
            return {
                'operations': {
                    name: metrics.to_dict()
                    for name, metrics in self._metrics.items()
                },
                'session': self._session.to_dict(),
                'recent': {
                    name: {
                        'count': len(times),
                        'avg_ms': sum(times) / len(times) if times else 0,
                    }
                    for name, times in self._recent_times.items()
                }
            }
    
    def get_percentile(self, operation: str, percentile: float) -> float:
        """Get percentile for operation time."""
        with self._lock:
            if operation not in self._recent_times:
                return 0
            
            times = sorted(self._recent_times[operation])
            idx = int(len(times) * percentile / 100)
            return times[min(idx, len(times) - 1)]
    
    def reset(self):
        """Reset all metrics."""
        with self._lock:
            self._metrics.clear()
            self._recent_times.clear()
            self._session = SessionStats()
    
    def export_json(self, path: str):
        """Export metrics to JSON file."""
        stats = self.get_stats()
        with open(path, 'w') as f:
            json.dump(stats, f, indent=2)
    
    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        stats = self.get_stats()
        
        for name, metrics in stats['operations'].items():
            prefix = f"turboquant_{name}"
            lines.append(f"# TYPE {prefix}_count counter")
            lines.append(f"{prefix}_count {metrics['count']}")
            lines.append(f"# TYPE {prefix}_duration_ms summary")
            lines.append(f"{prefix}_duration_ms_sum {metrics['total_time_ms']}")
            lines.append(f"{prefix}_duration_ms_count {metrics['count']}")
        
        return '\n'.join(lines)


class TurboQuantLogger:
    """
    Logging utility for TurboQuant operations.
    
    Features:
      - Structured logging
      - Performance warnings
      - Memory alerts
      - Quality degradation alerts
    """
    
    def __init__(
        self,
        name: str = "turboquant",
        level: str = "INFO",
        log_file: Optional[str] = None
    ):
        """
        Initialize logger.
        
        Args:
            name: Logger name
            level: Log level
            log_file: Optional file to log to
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(console_handler)
        
        # File handler (optional)
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            self.logger.addHandler(file_handler)
        
        self._metrics = MetricsCollector()
    
    @contextmanager
    def log_operation(self, name: str, log_level: str = "DEBUG"):
        """
        Context manager with logging.
        
        Usage:
            with logger.log_operation("encode"):
                encoded = codec.encode_keys_batch(keys)
        """
        self.logger.log(getattr(logging, log_level.upper()), f"Starting {name}")
        start_time = time.perf_counter()
        
        try:
            with self._metrics.track_operation(name):
                yield
            elapsed = (time.perf_counter() - start_time) * 1000
            self.logger.log(
                getattr(logging, log_level.upper()),
                f"Completed {name} in {elapsed:.2f}ms"
            )
        except Exception as e:
            self.logger.error(f"Failed {name}: {str(e)}")
            raise
    
    def log_compression(
        self,
        original_mb: float,
        compressed_mb: float,
        ratio: float
    ):
        """Log compression statistics."""
        savings_mb = original_mb - compressed_mb
        savings_pct = (1 - ratio) * 100
        
        self.logger.info(
            f"Compression: {original_mb:.2f}MB → {compressed_mb:.2f}MB "
            f"(saved {savings_mb:.2f}MB, {savings_pct:.1f}%)"
        )
    
    def log_quality(
        self,
        metric_name: str,
        value: float,
        threshold: Optional[float] = None
    ):
        """
        Log quality metric with optional warning.
        
        Args:
            metric_name: Name of metric
            value: Metric value
            threshold: Warning threshold
        """
        if threshold is not None and value < threshold:
            self.logger.warning(
                f"Quality degradation: {metric_name}={value:.4f} "
                f"(threshold={threshold:.4f})"
            )
        else:
            self.logger.info(f"Quality: {metric_name}={value:.4f}")
    
    def log_memory(self, used_mb: float, limit_mb: Optional[float] = None):
        """
        Log memory usage with optional alert.
        
        Args:
            used_mb: Memory used in MB
            limit_mb: Memory limit
        """
        if limit_mb is not None:
            usage_pct = (used_mb / limit_mb) * 100
            if usage_pct > 90:
                self.logger.warning(
                    f"High memory usage: {used_mb:.2f}MB / {limit_mb:.2f}MB "
                    f"({usage_pct:.1f}%)"
                )
            else:
                self.logger.info(
                    f"Memory: {used_mb:.2f}MB / {limit_mb:.2f}MB ({usage_pct:.1f}%)"
                )
        else:
            self.logger.info(f"Memory used: {used_mb:.2f}MB")
    
    def get_metrics(self) -> MetricsCollector:
        """Get metrics collector."""
        return self._metrics


# Global logger instance
_global_logger: Optional[TurboQuantLogger] = None


def get_logger() -> TurboQuantLogger:
    """Get global logger instance."""
    global _global_logger
    if _global_logger is None:
        _global_logger = TurboQuantLogger()
    return _global_logger


def enable_logging(level: str = "INFO", log_file: Optional[str] = None):
    """Enable global logging."""
    global _global_logger
    _global_logger = TurboQuantLogger(level=level, log_file=log_file)
    return _global_logger


# Decorator for automatic tracking
def track_operation(name: str):
    """
    Decorator for tracking operation metrics.
    
    Usage:
        @track_operation("encode")
        def encode_keys(self, keys):
            return self.codec.encode_keys_batch(keys)
    """
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            logger = get_logger()
            with logger.log_operation(name):
                return func(*args, **kwargs)
        return wrapper
    return decorator
