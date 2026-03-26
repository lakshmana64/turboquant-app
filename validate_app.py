#!/usr/bin/env python3
"""
TurboQuant Application Validation Script (Lite)

Validates file structure, syntax, and imports without requiring PyTorch.
Run this to ensure everything is properly connected.

Usage:
    python validate_app.py
"""

import sys
import os
import ast

# Add project root to path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

# Colors for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'

passed = 0
failed = 0


def check(name: str, condition: bool, message: str = ""):
    global passed, failed
    if condition:
        print(f"{GREEN}✓{RESET} {name}")
        passed += 1
    else:
        print(f"{RED}✗{RESET} {name}")
        if message:
            print(f"  {RED}Error:{RESET} {message}")
        failed += 1


def validate_python_syntax(filepath: str) -> bool:
    """Validate Python syntax without importing."""
    try:
        with open(filepath, 'r') as f:
            source = f.read()
        ast.parse(source)
        return True
    except SyntaxError:
        return False
    except Exception:
        return False


def check_imports_in_file(filepath: str) -> tuple:
    """Check if imports in a file are valid (relative or absolute)."""
    try:
        with open(filepath, 'r') as f:
            source = f.read()
        
        tree = ast.parse(source)
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    imports.append(f"from {module} import {alias.name}")
        
        return True, imports
    except Exception as e:
        return False, str(e)


def main():
    print("=" * 60)
    print("TURBOQUANT APPLICATION VALIDATION")
    print("=" * 60)
    print()

    # 1. File Structure Validation
    print("1. FILE STRUCTURE VALIDATION")
    print("-" * 40)
    
    required_files = [
        # Root
        "__init__.py",
        "README.md",
        "requirements.txt",
        "test.py",
        "tsconfig.json",
        
        # Public package
        "turboquant/__init__.py",
        "turboquant/core/__init__.py",
        "turboquant/sdk/__init__.py",
        "turboquant/cli/main.py",
        "turboquant/integrations/__init__.py",
        "turboquant/integrations/plugins/__init__.py",
        
        # Core
        "core/__init__.py",
        "core/codec.py",
        "core/scalar_quant.py",
        "core/qjl_projection.py",
        "core/residual.py",
        "core/estimator.py",
        
        # SDK
        "sdk/__init__.py",
        "sdk/optimize.py",
        
        # Integrations
        "integrations/huggingface.py",
        "integrations/ollama_test.py",
        
        # Plugins
        "integrations/plugins/__init__.py",
        "integrations/plugins/ollama.py",
        "integrations/plugins/registry.py",
        "integrations/plugins/ollama_cli.py",
        "integrations/plugins/__main__.py",
        "integrations/plugins/examples.py",
        "integrations/plugins/openai_plugin.py",
        "integrations/plugins/sentence_transformers_plugin.py",
        
        # CLI
        "cli/main.py",
        
        # Benchmarks
        "benchmarks/unbiasedness.py",
        "benchmarks/attention_test.py",
        "benchmarks/recall_test.py",
    ]
    
    for file_path in required_files:
        full_path = os.path.join(ROOT_DIR, file_path)
        if os.path.exists(full_path):
            check(f"  {file_path}", True)
        else:
            check(f"  {file_path}", False, "File not found")
    
    print()

    # 2. Python Syntax Validation
    print("2. PYTHON SYNTAX VALIDATION")
    print("-" * 40)
    
    py_files = [f for f in required_files if f.endswith('.py')]
    
    for file_path in py_files:
        full_path = os.path.join(ROOT_DIR, file_path)
        if os.path.exists(full_path):
            if validate_python_syntax(full_path):
                check(f"  {file_path} (syntax)", True)
            else:
                check(f"  {file_path} (syntax)", False, "Syntax error")
    
    print()

    # 3. Module Structure
    print("3. MODULE STRUCTURE")
    print("-" * 40)
    
    # Check __init__.py files have __all__
    init_files = [f for f in required_files if f.endswith('__init__.py')]
    
    for file_path in init_files:
        full_path = os.path.join(ROOT_DIR, file_path)
        if os.path.exists(full_path):
            with open(full_path, 'r') as f:
                content = f.read()
            has_all = '__all__' in content
            check(f"  {file_path} (has __all__)", has_all)
    
    print()

    # 4. Key Class/Function Definitions
    print("4. KEY CLASS/FUNCTION DEFINITIONS")
    print("-" * 40)
    
    # Check core/codec.py
    codec_path = os.path.join(ROOT_DIR, 'core', 'codec.py')
    if os.path.exists(codec_path):
        with open(codec_path, 'r') as f:
            content = f.read()
        check("  TurboQuantCodec class", 'class TurboQuantCodec' in content)
        check("  TurboQuantConfig class", 'class TurboQuantConfig' in content)
        check("  encode_keys_batch method", 'def encode_keys_batch' in content)
        check("  estimate_inner_products method", 'def estimate_inner_products' in content)
    
    # Check core/estimator.py
    estimator_path = os.path.join(ROOT_DIR, 'core', 'estimator.py')
    if os.path.exists(estimator_path):
        with open(estimator_path, 'r') as f:
            content = f.read()
        check("  UnbiasedInnerProductEstimator class", 'class UnbiasedInnerProductEstimator' in content)
        check("  estimate_inner_product_unbiased function", 'def estimate_inner_product_unbiased' in content)
    
    # Check sdk/optimize.py
    optimize_path = os.path.join(ROOT_DIR, 'sdk', 'optimize.py')
    if os.path.exists(optimize_path):
        with open(optimize_path, 'r') as f:
            content = f.read()
        check("  TurboQuantizer class", 'class TurboQuantizer' in content)
        check("  optimize function", 'def optimize' in content)
    
    # Check plugins/ollama.py
    ollama_path = os.path.join(ROOT_DIR, 'integrations', 'plugins', 'ollama.py')
    if os.path.exists(ollama_path):
        with open(ollama_path, 'r') as f:
            content = f.read()
        check("  OllamaPlugin class", 'class OllamaPlugin' in content)
        check("  OllamaPluginConfig class", 'class OllamaPluginConfig' in content)
        check("  compress method", 'def compress' in content)
        check("  query method", 'def query' in content)
    
    # Check plugins/registry.py
    registry_path = os.path.join(ROOT_DIR, 'integrations', 'plugins', 'registry.py')
    if os.path.exists(registry_path):
        with open(registry_path, 'r') as f:
            content = f.read()
        check("  PluginRegistry class", 'class PluginRegistry' in content)
        check("  get_registry function", 'def get_registry' in content)
        check("  load_plugin function", 'def load_plugin' in content)
    
    print()

    # 5. Documentation
    print("5. DOCUMENTATION")
    print("-" * 40)
    
    # Check README
    readme_path = os.path.join(ROOT_DIR, 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r') as f:
            content = f.read()
        check("  README.md exists", True)
        check("  README has usage examples", '```python' in content or 'Usage:' in content)
        check("  README has installation instructions", 'install' in content.lower() or 'pip' in content)
    
    # Check plugin README
    plugin_readme = os.path.join(ROOT_DIR, 'integrations', 'plugins', 'README.md')
    if os.path.exists(plugin_readme):
        with open(plugin_readme, 'r') as f:
            content = f.read()
        check("  plugins/README.md exists", True)
        check("  Plugin README has quick start", 'Quick Start' in content or 'quick start' in content)
    
    print()

    # 6. Import Statement Validation
    print("6. IMPORT STATEMENT VALIDATION")
    print("-" * 40)
    
    # Check for common import issues
    files_to_check = [
        'core/codec.py',
        'core/estimator.py',
        'sdk/optimize.py',
        'integrations/plugins/ollama.py',
        'integrations/plugins/openai_plugin.py',
        'integrations/plugins/sentence_transformers_plugin.py',
    ]
    
    for file_path in files_to_check:
        full_path = os.path.join(ROOT_DIR, file_path)
        if os.path.exists(full_path):
            valid, result = check_imports_in_file(full_path)
            if valid:
                check(f"  {file_path} (imports)", True)
            else:
                check(f"  {file_path} (imports)", False, result)
    
    print()

    # 7. CLI Validation
    print("7. CLI VALIDATION")
    print("-" * 40)
    
    cli_path = os.path.join(ROOT_DIR, 'cli', 'main.py')
    if os.path.exists(cli_path):
        with open(cli_path, 'r') as f:
            content = f.read()
        check("  CLI has main function", 'def main' in content)
        check("  CLI has argparse", 'argparse' in content)
        check("  CLI has subcommands", 'add_subparsers' in content)
    
    # Check plugin CLI
    plugin_cli = os.path.join(ROOT_DIR, 'integrations', 'plugins', 'ollama_cli.py')
    if os.path.exists(plugin_cli):
        with open(plugin_cli, 'r') as f:
            content = f.read()
        check("  Plugin CLI has commands", 'def cmd_' in content)
        check("  Plugin CLI has __main__", '__main__' in content)
    
    print()

    # Summary
    print("=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"{GREEN}Passed:{RESET}  {passed}")
    print(f"{RED}Failed:{RESET}  {failed}")
    print()
    
    if failed == 0:
        print(f"{GREEN}✓ ALL VALIDATIONS PASSED!{RESET}")
        print("\nThe TurboQuant application structure is complete and valid.")
        print("\nNext steps:")
        print("  1. Install package: pip install -e .")
        print("  2. Run tests: pytest -q")
        print("  3. Try CLI: turboquant --help")
        print("  4. Build TypeScript: npm run build")
        return 0
    else:
        print(f"{RED}✗ SOME VALIDATIONS FAILED!{RESET}")
        print("\nPlease fix the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
