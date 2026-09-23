# torch-mps-copy-dtype-guard

This repository has been consolidated into **torch-correctness-guards**.

Use the maintained umbrella package instead:

```bash
python -m pip install git+https://github.com/zhuhroscar-tech/torch-correctness-guards.git
python -m torch_correctness_guards.cli run mps-copy-dtype --json
```

Python API:

```python
from torch_correctness_guards import safe_to, safe_copy_
```

The migrated guard still checks the same PyTorch/MPS issue: Apple Silicon MPS-to-CPU copies into `float64` or `complex128` destinations can silently lose data. The umbrella package now contains the maintained implementation, CLI, and tests:

<https://github.com/zhuhroscar-tech/torch-correctness-guards>

This source repository is archived for history only.
