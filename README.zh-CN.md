# torch-mps-copy-dtype-guard

本仓库已合并进 **torch-correctness-guards**。

请改用维护中的总包：

```bash
python -m pip install git+https://github.com/zhuhroscar-tech/torch-correctness-guards.git
python -m torch_correctness_guards.cli run mps-copy-dtype --json
```

Python API：

```python
from torch_correctness_guards import safe_to, safe_copy_
```

迁移后的 guard 仍然检查同一个 PyTorch/MPS 问题：Apple Silicon 上 MPS 到 CPU 的 `float64` 或 `complex128` 目标拷贝可能静默丢失数据。现在维护中的实现、CLI 与测试都在总包中：

<https://github.com/zhuhroscar-tech/torch-correctness-guards>

本源仓库仅作为历史记录归档。
