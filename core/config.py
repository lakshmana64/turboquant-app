"""
TurboQuant Configuration & Environment Management.
Handles hardware detection and user preferences.
"""

import os
import json
import torch
from typing import Dict, Any

CONFIG_PATH = "turboquant_config.json"

def detect_environment() -> Dict[str, Any]:
    """Detect hardware and software capabilities."""
    env = {
        "has_cuda": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
        "has_triton": False,
        "platform": os.name
    }
    
    try:
        import triton
        env["has_triton"] = True
    except ImportError:
        pass
        
    return env

def save_user_config(config: Dict[str, Any]):
    """Save user preferences to disk."""
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=4)

def load_user_config() -> Dict[str, Any]:
    """Load user preferences."""
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    return {"mode": "standard", "use_triton": False}

def run_setup_wizard():
    """Interactive CLI wizard to configure TurboQuant."""
    print("\n🚀 TurboQuant Smart Setup")
    print("==========================")
    
    env = detect_environment()
    config = {}
    
    print(f"Hardware Detected: {'GPU (' + env['gpu_name'] + ')' if env['has_cuda'] else 'CPU Only'}")
    
    if env['has_cuda']:
        print("\n[PROMPT] We detected a compatible GPU environment!")
        use_triton = input("Would you like to enable Triton High-Performance kernels? (y/n) [y]: ").lower() != 'n'
        config["use_triton"] = use_triton
        config["mode"] = "turbo" if use_triton else "standard"
        
        if use_triton and not env['has_triton']:
            print("Note: You will need to install triton: 'pip install triton'")
    else:
        print("\n[INFO] Running in CPU mode. Standard optimizations will be used.")
        config["use_triton"] = False
        config["mode"] = "standard"
        
    print("\n[PROMPT] Would you like to enable the FastAPI Microservice for remote access?")
    config["enable_api"] = input("(y/n) [n]: ").lower() == 'y'
    
    save_user_config(config)
    print(f"\n✅ Configuration saved to {CONFIG_PATH}")
    print(f"Selected Mode: {config['mode'].upper()}")
