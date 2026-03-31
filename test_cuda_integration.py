#!/usr/bin/env python3
"""
Test CUDA Integration for TurboQuant + llama.cpp

This script tests the CUDA-accelerated llama.cpp integration with TurboQuant features.

Usage:
    python test_cuda_integration.py --model llama3:8b --gpu-layers 32
"""

import torch
import time
import argparse
import subprocess
import sys
from typing import Dict, Any, Optional


def check_cuda_available() -> bool:
    """Check if CUDA is available on the system."""
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            timeout=5
        )
        return result.returncode == 0
    except Exception:
        return False


def check_cuda_version() -> Optional[str]:
    """Get CUDA version."""
    try:
        result = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'release' in line:
                    return line.split('release')[1].split(',')[0].strip()
    except Exception:
        return None
    return None


def check_llama_cpp_cuda() -> bool:
    """Check if llama.cpp with CUDA support is installed."""
    import os
    llama_cpp_main = "./llama.cpp/main"
    
    if not os.path.exists(llama_cpp_main):
        return False
    
    try:
        result = subprocess.run(
            [llama_cpp_main, "--help"],
            capture_output=True,
            timeout=5
        )
        # Check if CUDA is mentioned in help
        return "gpu-layers" in result.stdout.decode()
    except Exception:
        return False


def test_cuda_inference(
    model_path: str,
    gpu_layers: int = 32,
    prompt: str = "Explain CUDA acceleration",
    max_tokens: int = 64
) -> Dict[str, Any]:
    """
    Test CUDA-accelerated inference.
    
    Args:
        model_path: Path to GGUF model
        gpu_layers: Number of layers to offload to GPU
        prompt: Input prompt
        max_tokens: Maximum tokens to generate
    
    Returns:
        Dictionary with inference results
    """
    import os
    llama_cpp_main = "./llama.cpp/main"
    
    if not os.path.exists(llama_cpp_main):
        return {
            "success": False,
            "error": "llama.cpp main binary not found"
        }
    
    cmd = [
        llama_cpp_main,
        "-m", model_path,
        "-p", prompt,
        "-n", str(max_tokens),
        "--gpu-layers", str(gpu_layers),
        "-t", str(torch.cuda.device_count() * 8),  # Use all CPU threads
        "--temp", "0.7",
        "--top-p", "0.95"
    ]
    
    start = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120
        )
        
        latency = time.time() - start
        
        if result.returncode == 0:
            # Parse output for tokens/second
            output = result.stdout
            tokens_per_sec = 0
            
            for line in output.split('\n'):
                if 'tokens/second' in line.lower():
                    try:
                        tokens_per_sec = float(line.split(':')[1].strip().split()[0])
                    except:
                        pass
            
            return {
                "success": True,
                "output": output,
                "latency_seconds": latency,
                "tokens_per_second": tokens_per_sec,
                "gpu_layers": gpu_layers
            }
        else:
            return {
                "success": False,
                "error": result.stderr
            }
            
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": "Inference timed out"
        }


def test_turboquant_kv_cache(
    model_path: str,
    gpu_layers: int = 32,
    k_type: str = "q8_0",
    v_type: str = "turbo4"
) -> Dict[str, Any]:
    """
    Test TurboQuant KV cache with CUDA.
    
    Args:
        model_path: Path to GGUF model
        gpu_layers: Number of GPU layers
        k_type: KV cache type for Keys
        v_type: KV cache type for Values
    
    Returns:
        Dictionary with test results
    """
    import os
    llama_cpp_main = "./llama.cpp/main"
    
    if not os.path.exists(llama_cpp_main):
        return {
            "success": False,
            "error": "llama.cpp main binary not found"
        }
    
    cmd = [
        llama_cpp_main,
        "-m", model_path,
        "-p", "Explain quantization",
        "-n", "64",
        "--gpu-layers", str(gpu_layers),
        "--kv-cache-type-k", k_type,
        "--kv-cache-type-v", v_type,
        "-c", "4096"
    ]
    
    start = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120
        )
        
        latency = time.time() - start
        
        if result.returncode == 0:
            return {
                "success": True,
                "output": result.stdout,
                "latency_seconds": latency,
                "k_cache_type": k_type,
                "v_cache_type": v_type,
                "context_size": 4096
            }
        else:
            return {
                "success": False,
                "error": result.stderr
            }
            
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": "Test timed out"
        }


def benchmark_cuda_vs_cpu(
    model_path: str,
    prompt: str = "Test prompt",
    max_tokens: int = 64
) -> Dict[str, Any]:
    """
    Benchmark CUDA vs CPU performance.
    
    Args:
        model_path: Path to model
        prompt: Test prompt
        max_tokens: Tokens to generate
    
    Returns:
        Benchmark comparison results
    """
    import os
    llama_cpp_main = "./llama.cpp/main"
    
    results = {}
    
    # CPU-only test
    print("Running CPU-only benchmark...")
    cmd_cpu = [
        llama_cpp_main,
        "-m", model_path,
        "-p", prompt,
        "-n", str(max_tokens),
        "--gpu-layers", "0",
        "-t", "8"
    ]
    
    start = time.time()
    result_cpu = subprocess.run(cmd_cpu, capture_output=True, text=True, timeout=120)
    cpu_time = time.time() - start
    
    if result_cpu.returncode == 0:
        results["cpu"] = {
            "time_seconds": cpu_time,
            "tokens_per_second": max_tokens / cpu_time if cpu_time > 0 else 0
        }
    
    # CUDA test
    print("Running CUDA benchmark...")
    cmd_cuda = [
        llama_cpp_main,
        "-m", model_path,
        "-p", prompt,
        "-n", str(max_tokens),
        "--gpu-layers", "32",
        "-t", "8"
    ]
    
    start = time.time()
    result_cuda = subprocess.run(cmd_cuda, capture_output=True, text=True, timeout=120)
    cuda_time = time.time() - start
    
    if result_cuda.returncode == 0:
        results["cuda"] = {
            "time_seconds": cuda_time,
            "tokens_per_second": max_tokens / cuda_time if cuda_time > 0 else 0
        }
    
    # Calculate speedup
    if "cpu" in results and "cuda" in results:
        speedup = results["cpu"]["time_seconds"] / results["cuda"]["time_seconds"]
        results["speedup"] = speedup
        results["cuda_faster_by"] = f"{speedup:.1f}x"
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Test CUDA integration for TurboQuant")
    parser.add_argument("--model", type=str, default="llama3:8b", help="Model name or path")
    parser.add_argument("--gpu-layers", type=int, default=32, help="Number of GPU layers")
    parser.add_argument("--prompt", type=str, default="Explain CUDA acceleration in TurboQuant", help="Test prompt")
    parser.add_argument("--max-tokens", type=int, default=64, help="Max tokens to generate")
    parser.add_argument("--benchmark", action="store_true", help="Run CUDA vs CPU benchmark")
    
    args = parser.parse_args()
    
    print("="*70)
    print("TurboQuant CUDA Integration Test")
    print("="*70)
    print()
    
    # Check CUDA
    print("Checking CUDA availability...")
    if check_cuda_available():
        print("✓ CUDA is available")
        cuda_version = check_cuda_version()
        if cuda_version:
            print(f"  CUDA Version: {cuda_version}")
        
        # Get GPU info
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                for i, line in enumerate(result.stdout.strip().split('\n')):
                    gpu_name, memory = line.split(', ')
                    print(f"  GPU {i}: {gpu_name} ({memory})")
        except:
            pass
    else:
        print("✗ CUDA not available")
        print("  Please install CUDA Toolkit and ensure NVIDIA GPU is present")
        return
    
    print()
    
    # Check llama.cpp
    print("Checking llama.cpp with CUDA...")
    if check_llama_cpp_cuda():
        print("✓ llama.cpp with CUDA support found")
    else:
        print("✗ llama.cpp with CUDA not found")
        print("  Run: bash build_llama_cpp_cuda.sh")
        return
    
    print()
    
    # Test basic inference
    print(f"Testing CUDA inference with model: {args.model}")
    print(f"  GPU Layers: {args.gpu_layers}")
    print(f"  Prompt: {args.prompt[:50]}...")
    print()
    
    result = test_cuda_inference(
        model_path=args.model,
        gpu_layers=args.gpu_layers,
        prompt=args.prompt,
        max_tokens=args.max_tokens
    )
    
    if result.get("success"):
        print("✓ CUDA inference successful")
        print(f"  Latency: {result['latency_seconds']:.2f}s")
        print(f"  Tokens/sec: {result['tokens_per_second']:.1f}")
        print(f"  Output preview: {result['output'][:200]}...")
    else:
        print(f"✗ CUDA inference failed: {result.get('error')}")
    
    print()
    
    # Test TurboQuant KV cache
    print("Testing TurboQuant KV cache with CUDA...")
    tq_result = test_turboquant_kv_cache(
        model_path=args.model,
        gpu_layers=args.gpu_layers,
        k_type="q8_0",
        v_type="turbo4"
    )
    
    if tq_result.get("success"):
        print("✓ TurboQuant KV cache test successful")
        print(f"  K Cache: q8_0")
        print(f"  V Cache: turbo4")
        print(f"  Latency: {tq_result['latency_seconds']:.2f}s")
    else:
        print(f"✗ TurboQuant KV cache test failed: {tq_result.get('error')}")
    
    print()
    
    # Run benchmark if requested
    if args.benchmark:
        print("Running CUDA vs CPU benchmark...")
        print()
        
        bench_result = benchmark_cuda_vs_cpu(
            model_path=args.model,
            prompt=args.prompt,
            max_tokens=args.max_tokens
        )
        
        if "cpu" in bench_result and "cuda" in bench_result:
            print("Benchmark Results:")
            print(f"  CPU:  {bench_result['cpu']['tokens_per_second']:.1f} tokens/sec")
            print(f"  CUDA: {bench_result['cuda']['tokens_per_second']:.1f} tokens/sec")
            print(f"  Speedup: {bench_result['cuda_faster_by']} faster")
        else:
            print("✗ Benchmark failed")
    
    print()
    print("="*70)
    print("CUDA Integration Test Complete")
    print("="*70)


if __name__ == "__main__":
    main()
