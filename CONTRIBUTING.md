# Contributing to TurboQuant

Thank you for your interest in contributing to TurboQuant! We welcome contributions from the community to help make online vector quantization even better.

## 🚀 Getting Started

1. **Fork the repository** on GitHub.
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/tonbistudio/turboquant-pytorch.git
   cd turboquant-pytorch
   ```
3. **Create a virtual environment** and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -e .[test,dev,plugins]
   ```

## 🛠️ Development Workflow

### 1. Create a Branch
Create a new branch for your feature or bug fix:
```bash
git checkout -b feat/your-feature-name
# OR
git checkout -b fix/your-bug-fix
```

### 2. Make Your Changes
Ensure your code follows our standards:
- Use type hints for all functions.
- Add docstrings in Google style.
- Maintain consistency with the existing codebase.

### 3. Run Tests
Before submitting, ensure all tests pass:
```bash
pytest test.py
```

### 4. Linting and Formatting
We use `ruff` for linting and formatting. Run:
```bash
ruff check .
ruff format .
```

## 📬 Submitting a Pull Request

1. **Push your changes** to your fork.
2. **Open a Pull Request** against the `main` branch.
3. **Describe your changes** clearly in the PR template.
4. **Wait for review**. We aim to review all PRs within a few days.

## 🐛 Reporting Bugs
If you find a bug, please use the **Bug Report** issue template and include:
- A clear description of the issue.
- Steps to reproduce the bug.
- Your environment details (OS, Python version, PyTorch version).

## 💡 Feature Requests
We love new ideas! Please use the **Feature Request** issue template to suggest enhancements.

## 📄 License
By contributing, you agree that your contributions will be licensed under the project's MIT License.
