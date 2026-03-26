from setuptools import setup, find_packages

setup(
    name="turboquant",
    version="0.1.0",
    description="TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate",
    author="TurboQuant Contributors",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "requests",
        "pandas",
        "pyyaml",
    ],
    extras_require={
        "app": [
            "gradio>=6.0.0",
            "plotly>=5.0.0",
        ],
        "plugins": [
            "sentence-transformers>=3.0.0",
        ],
        "test": ["pytest"],
        "dev": [
            "pytest",
            "gradio>=6.0.0",
            "plotly>=5.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "turboquant=turboquant.cli.main:main",
        ],
    },
    python_requires=">=3.8",
)
