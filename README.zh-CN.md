# torch-mps-copy-dtype-guard

修复 PyTorch 近期版本中一个真实可复现的正确性缺陷：将 MPS
（Apple Silicon GPU）张量通过 `.to()` 或 `.copy_()` 传输到 CPU 的
`float64` 或 `complex128` 目标时，会**静默写入零值**（若目标是预先
填充好的 `.copy_()` 目标，则**完全不写入、保持原值不变**），而不是
抛出错误或正确转换。没有异常，没有警告——只是数据错了。

上游 issue：[pytorch/pytorch#197715](https://github.com/pytorch/pytorch/issues/197715)
（"[MPS] Copy from MPS into a CPU float64 or complex128 tensor
silently writes nothing (2.14 regression)"），本仓库创建时
（2026-09-20）仍为 open 状态，标记为 `module: correctness (silent)`
和 `module: mps`。相关 PR [#197722](https://github.com/pytorch/pytorch/pull/197722)
让失败变得"响亮"（改为抛错而非静默空操作），但本身并不提供*正确的
转换结果*——且截至本文撰写时尚未合并。

## 为什么重要

根本原因（根据 issue 报告者，并与 PyTorch 的 `c10`/Metal 索引代码一致）：
MPS 复制/转换内核使用的 Metal `store_at_offs` switch 语句没有覆盖
`Double`/`ComplexDouble` 目标类型的分支，导致这两种 dtype 的写入操作
静默地变成空操作。

任何在 macOS/Apple Silicon 上将 MPS 张量搬到 CPU 做 float64 精度后处理
（NumPy 互操作、科学计算归约、复数信号处理）或通过 `copy_()` 写入预分配
float64/complex128 缓冲区的工作流，都会静默得到错误数据，**没有任何
信号提示出了问题**。

## 在真实硬件上独立复现

本 bug 在本仓库自己的构建/测试主机上从零复现：**Apple M4，macOS
15.7.7，torch 2.14.0，真实（非虚拟化）MPS 设备**——不是模拟的，也不是
仅凭 issue 文字假设的。

```
>>> x = torch.arange(1, 9, dtype=torch.float32, device="mps") / 3
>>> x.to("cpu", torch.float64)
tensor([0., 0., 0., 0., 0., 0., 0., 0.], dtype=torch.float64)   # 错误：应为 [0.3333, 0.6667, ...]

>>> dst = torch.full((8,), 7.0, dtype=torch.float64)
>>> dst.copy_(x)
>>> dst
tensor([7., 7., 7., 7., 7., 7., 7., 7.], dtype=torch.float64)   # 错误：dst 完全未被修改
```

## 修复方案

`safe_to()` 和 `safe_copy_()` 是 `.to()`/`.copy_()` 的替代函数，
彻底绕开该缺陷：对于这两种受影响的 dtype 且源为 MPS 的组合，先按照
**源张量自身的 dtype** 完成传输（Metal 对此始终处理正确），再在 CPU
侧进行第二步的类型提升到 `float64`/`complex128`。其他所有组合都直接
委托给原生操作，不对已经正确的路径做多余的保护。

```python
from torch_mps_copy_dtype_guard import safe_to, safe_copy_

x = torch.arange(1, 9, dtype=torch.float32, device="mps") / 3
y = safe_to(x, "cpu", torch.float64)          # 正确：[0.3333, 0.6667, ...]

dst = torch.full((8,), 7.0, dtype=torch.float64)
safe_copy_(dst, x)                             # 正确：dst 现在保存转换后的值
```

## 命令行工具

```
$ pip install -e '.[torch]'
$ torch-mps-copy-dtype-guard
```

`--json` 输出机器可读结果，`--no-color` 关闭 ANSI 颜色（同时支持
`NO_COLOR`/`FORCE_COLOR` 环境变量）。

## 诚实的局限性

- **仅保护通过本模块函数调用的 `.to()`/`.copy_()`。** 不会全局
  monkeypatch `torch.Tensor.to`/`copy_`——需要在真正搬运 MPS 张量到
  CPU float64/complex128 目标的调用处显式调用 `safe_to`/`safe_copy_`。
- **仅针对 MPS。** 在没有真实可用 MPS 设备的主机上（包括那些报告
  `torch.backends.mps.is_available() == True` 但实际无法分配显存的
  虚拟化 CI 运行器——参见 `actions/runner-images#9918`），MPS 相关
  测试会被诚实地跳过，而不是静默通过。本仓库自己的 CI 无法执行 MPS
  路径（GitHub Actions 没有真正的 Apple Silicon GPU 运行器）；上面的
  MPS 复现与修复验证是在真实 Apple Silicon 硬件上运行的，而非 CI。
  `safe_to`/`safe_copy_` 的纯 CPU 用法是可移植的，并在 CI 的
  `ubuntu-latest` 与 `macos-latest` 上都经过测试。
- **针对特定上游回归。** 如果 PyTorch 上游修复了
  `pytorch/pytorch#197715`，"复现原生 bug" 的测试会在拥有可用 MPS 的
  主机上开始失败并给出清晰提示——这是需要重新核实上游状态的信号，
  不是本修复模块本身的缺陷。无论底层 bug 是否仍存在，修复函数在两种
  情况下调用都是安全的（结果始终正确）。

## 安装

```
pip install -e '.[dev,torch]'   # 开发/测试用
pip install -e '.[torch]'       # 使用
```

## 许可证

MIT
