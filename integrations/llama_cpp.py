"""
llama.cpp Integration for TurboQuant.

Provides integration with llama.cpp for production LLM inference with:
- KV cache compression using TurboQuant formats
- Metal GPU kernels (Apple Silicon)
- CUDA support (NVIDIA GPUs)
- GGUF model quantization support

Requirements:
- llama.cpp fork with TurboQuant support
- GGUF models with TurboQuant KV cache

Reference:
- https://github.com/TheTom/turboquant_plus (llama.cpp integration)
- https://github.com/ggerganov/llama.cpp
"""

import os
import subprocess
import json
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from pathlib import Path


@dataclass
class LlamaCppConfig:
    """Configuration for llama.cpp integration."""
    
    llama_cpp_path: str = "./llama.cpp"
    model_path: str = ""
    kv_cache_type_k: str = "q8_0"  # Key cache type
    kv_cache_type_v: str = "turbo4"  # Value cache type
    context_size: int = 4096
    batch_size: int = 512
    threads: int = 4
    use_metal: bool = False
    use_cuda: bool = False
    verbose: bool = False
    
    def __post_init__(self):
        """Validate configuration."""
        if self.use_metal and self.use_cuda:
            raise ValueError("Cannot use both Metal and CUDA")
        
        # Validate KV cache types
        valid_types = ["q8_0", "q4_0", "turbo2", "turbo3", "turbo4"]
        if self.kv_cache_type_k not in valid_types:
            raise ValueError(f"Invalid K cache type: {self.kv_cache_type_k}")
        if self.kv_cache_type_v not in valid_types:
            raise ValueError(f"Invalid V cache type: {self.kv_cache_type_v}")


class LlamaCppIntegration:
    """
    Integration layer for llama.cpp with TurboQuant support.
    
    Provides Python interface to llama.cpp with TurboQuant KV cache
    compression for production LLM inference.
    """
    
    def __init__(self, config: LlamaCppConfig):
        """
        Initialize llama.cpp integration.
        
        Args:
            config: LlamaCppConfig instance
        """
        self.config = config
        self.llama_cpp_path = Path(config.llama_cpp_path)
        self.main_binary = self.llama_cpp_path / "main"
        self.server_binary = self.llama_cpp_path / "server"
        
        # Check if llama.cpp is built
        self.is_available = self._check_installation()
        
        # Process handle for server mode
        self.server_process: Optional[subprocess.Popen] = None
        self.server_url: str = "http://localhost:8080"
    
    def _check_installation(self) -> bool:
        """Check if llama.cpp is properly installed."""
        if not self.llama_cpp_path.exists():
            print(f"Warning: llama.cpp not found at {self.llama_cpp_path}")
            return False
        
        if not self.main_binary.exists():
            print(f"Warning: llama.cpp main binary not found")
            print("Please build llama.cpp first:")
            print("  cd llama.cpp && mkdir build && cd build && cmake .. && make")
            return False
        
        return True
    
    def build_llama_cpp(
        self,
        use_metal: Optional[bool] = None,
        use_cuda: Optional[bool] = None,
        clean: bool = False
    ) -> bool:
        """
        Build llama.cpp with appropriate backend.
        
        Args:
            use_metal: Enable Metal backend (Apple Silicon)
            use_cuda: Enable CUDA backend (NVIDIA)
            clean: Clean build directory first
        
        Returns:
            True if build succeeded
        """
        use_metal = use_metal if use_metal is not None else self.config.use_metal
        use_cuda = use_cuda if use_cuda is not None else self.config.use_cuda
        
        build_dir = self.llama_cpp_path / "build"
        build_dir.mkdir(exist_ok=True)
        
        if clean and build_dir.exists():
            import shutil
            shutil.rmtree(build_dir)
            build_dir.mkdir()
        
        # Build command
        cmake_args = ["cmake", ".."]
        
        if use_metal:
            cmake_args.extend(["-DGGML_METAL=ON"])
            print("Building with Metal backend (Apple Silicon)")
        elif use_cuda:
            cmake_args.extend(["-DGGML_CUDA=ON"])
            print("Building with CUDA backend (NVIDIA)")
        else:
            print("Building with CPU backend")
        
        # Run cmake
        try:
            subprocess.run(
                cmake_args,
                cwd=build_dir,
                check=True,
                capture_output=not self.config.verbose
            )
            subprocess.run(
                ["cmake", "--build", ".", "--config", "Release"],
                cwd=build_dir,
                check=True,
                capture_output=not self.config.verbose
            )
            
            # Copy binaries to root
            import shutil
            for binary in ["main", "server", "quantize"]:
                src = build_dir / "bin" / binary
                dst = self.llama_cpp_path / binary
                if src.exists():
                    shutil.copy2(src, dst)
            
            print("✓ llama.cpp built successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"✗ Build failed: {e}")
            return False
    
    def run_inference(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.8,
        top_p: float = 0.95
    ) -> Dict[str, Any]:
        """
        Run inference using llama.cpp main binary.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
        
        Returns:
            Dictionary with generation results
        """
        if not self.is_available:
            raise RuntimeError("llama.cpp is not available")
        
        if not self.config.model_path:
            raise ValueError("Model path not specified")
        
        # Build command
        cmd = [
            str(self.main_binary),
            "-m", self.config.model_path,
            "-n", str(max_tokens),
            "--temp", str(temperature),
            "--top-p", str(top_p),
            "-c", str(self.config.context_size),
            "-b", str(self.config.batch_size),
            "-t", str(self.config.threads),
            "--kv-cache-type-k", self.config.kv_cache_type_k,
            "--kv-cache-type-v", self.config.kv_cache_type_v,
            "-p", prompt
        ]
        
        if self.config.use_metal:
            cmd.append("-ngl")
            cmd.append("1")  # Offload to Metal
        
        # Run inference
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            return {
                "success": True,
                "output": result.stdout,
                "stderr": result.stderr
            }
            
        except subprocess.CalledProcessError as e:
            return {
                "success": False,
                "error": str(e),
                "stderr": e.stderr
            }
    
    def start_server(
        self,
        host: str = "localhost",
        port: int = 8080
    ) -> bool:
        """
        Start llama.cpp server for API access.
        
        Args:
            host: Server host
            port: Server port
        
        Returns:
            True if server started successfully
        """
        if not self.is_available:
            raise RuntimeError("llama.cpp is not available")
        
        self.server_url = f"http://{host}:{port}"
        
        # Build server command
        cmd = [
            str(self.server_binary),
            "-m", self.config.model_path,
            "-c", str(self.config.context_size),
            "-b", str(self.config.batch_size),
            "-t", str(self.config.threads),
            "--host", host,
            "--port", str(port),
            "--kv-cache-type-k", self.config.kv_cache_type_k,
            "--kv-cache-type-v", self.config.kv_cache_type_v
        ]
        
        if self.config.use_metal:
            cmd.extend(["-ngl", "1"])
        
        # Start server process
        try:
            self.server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait for server to start
            import time
            time.sleep(3)
            
            if self.server_process.poll() is not None:
                # Server failed to start
                stderr = self.server_process.stderr.read().decode()
                print(f"Server failed to start: {stderr}")
                return False
            
            print(f"✓ Server started at {self.server_url}")
            return True
            
        except Exception as e:
            print(f"✗ Failed to start server: {e}")
            return False
    
    def stop_server(self):
        """Stop the llama.cpp server."""
        if self.server_process is not None:
            self.server_process.terminate()
            self.server_process.wait()
            self.server_process = None
            print("Server stopped")
    
    def quantize_model(
        self,
        input_model: str,
        output_model: str,
        quantization_type: str = "Q4_K_M"
    ) -> Dict[str, Any]:
        """
        Quantize a model using llama.cpp quantize tool.
        
        Args:
            input_model: Path to input FP16 model
            output_model: Path for output quantized model
            quantization_type: Quantization type (Q4_K_M, Q8_0, etc.)
        
        Returns:
            Dictionary with quantization results
        """
        quantize_binary = self.llama_cpp_path / "quantize"
        
        if not quantize_binary.exists():
            return {
                "success": False,
                "error": "quantize binary not found"
            }
        
        cmd = [
            str(quantize_binary),
            input_model,
            output_model,
            quantization_type
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            return {
                "success": True,
                "output": result.stdout,
                "model_path": output_model,
                "quantization_type": quantization_type
            }
            
        except subprocess.CalledProcessError as e:
            return {
                "success": False,
                "error": str(e),
                "stderr": e.stderr
            }
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """
        Estimate memory usage with current KV cache settings.
        
        Returns:
            Dictionary with memory breakdown
        """
        # Approximate memory calculation
        # Based on llama.cpp memory accounting
        
        # KV cache memory per layer
        dim = 4096  # Typical hidden dim
        layers = 32  # Typical num layers
        heads = 32  # Typical num heads
        
        # Bits per value for different types
        type_bits = {
            "q8_0": 8,
            "q4_0": 4,
            "turbo2": 2.5,
            "turbo3": 3.5,
            "turbo4": 4.25
        }
        
        k_bits = type_bits.get(self.config.kv_cache_type_k, 8)
        v_bits = type_bits.get(self.config.kv_cache_type_v, 4)
        
        # Memory per token
        kv_memory_per_token = (
            (dim * k_bits + dim * v_bits) * layers / 8  # Bytes
        )
        
        # Total KV cache memory
        total_kv_memory = kv_memory_per_token * self.config.context_size
        
        return {
            "kv_cache_type_k": self.config.kv_cache_type_k,
            "kv_cache_type_v": self.config.kv_cache_type_v,
            "memory_per_token_mb": kv_memory_per_token / (1024 * 1024),
            "total_kv_cache_mb": total_kv_memory / (1024 * 1024),
            "context_size": self.config.context_size,
            "compression_vs_fp16": f"{16 / ((k_bits + v_bits) / 2):.1f}x"
        }
    
    def benchmark(
        self,
        num_tokens: int = 128,
        prompt: str = "The quick brown fox"
    ) -> Dict[str, Any]:
        """
        Run benchmark to measure performance.
        
        Args:
            num_tokens: Number of tokens to generate
            prompt: Prompt to use
        
        Returns:
            Benchmark results
        """
        import time
        
        # Warm-up
        self.run_inference(prompt, max_tokens=8)
        
        # Measure prefill
        start = time.time()
        result = self.run_inference(prompt, max_tokens=num_tokens)
        total_time = time.time() - start
        
        if not result.get("success"):
            return {"success": False, "error": result.get("error")}
        
        # Parse output for timing info
        output = result.get("output", "")
        
        return {
            "success": True,
            "total_time_s": total_time,
            "tokens_per_second": num_tokens / total_time,
            "memory_usage": self.get_memory_usage()
        }


def create_llama_cpp_integration(
    llama_cpp_path: str = "./llama.cpp",
    model_path: str = "",
    kv_cache_type_k: str = "q8_0",
    kv_cache_type_v: str = "turbo4",
    use_metal: bool = False,
    use_cuda: bool = False
) -> LlamaCppIntegration:
    """
    Factory function to create llama.cpp integration.
    
    Args:
        llama_cpp_path: Path to llama.cpp directory
        model_path: Path to GGUF model
        kv_cache_type_k: KV cache type for Keys
        kv_cache_type_v: KV cache type for Values
        use_metal: Enable Metal backend
        use_cuda: Enable CUDA backend
    
    Returns:
        Configured LlamaCppIntegration instance
    """
    config = LlamaCppConfig(
        llama_cpp_path=llama_cpp_path,
        model_path=model_path,
        kv_cache_type_k=kv_cache_type_k,
        kv_cache_type_v=kv_cache_type_v,
        use_metal=use_metal,
        use_cuda=use_cuda
    )
    
    return LlamaCppIntegration(config)


def check_turboquant_support() -> Dict[str, Any]:
    """
    Check if llama.cpp has TurboQuant support.
    
    Returns:
        Dictionary with support status
    """
    result = {
        "has_llama_cpp": False,
        "has_turboquant": False,
        "supported_formats": [],
        "backend": "unknown"
    }
    
    # Check for llama.cpp
    llama_cpp_paths = [
        Path("./llama.cpp"),
        Path.home() / "llama.cpp",
        Path("/opt/llama.cpp")
    ]
    
    for path in llama_cpp_paths:
        if path.exists():
            result["has_llama_cpp"] = True
            result["llama_cpp_path"] = str(path)
            break
    
    if not result["has_llama_cpp"]:
        return result
    
    # Check help output for TurboQuant options
    main_binary = Path(result["llama_cpp_path"]) / "main"
    if main_binary.exists():
        try:
            help_result = subprocess.run(
                [str(main_binary), "--help"],
                capture_output=True,
                text=True
            )
            
            help_text = help_result.stdout + help_result.stderr
            
            if "--kv-cache-type" in help_text or "turbo" in help_text.lower():
                result["has_turboquant"] = True
                result["supported_formats"] = ["q8_0", "q4_0", "turbo2", "turbo3", "turbo4"]
            
            if "metal" in help_text.lower():
                result["backend"] = "metal"
            elif "cuda" in help_text.lower():
                result["backend"] = "cuda"
            else:
                result["backend"] = "cpu"
                
        except Exception:
            pass
    
    return result
