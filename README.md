# torch-mps-copy-dtype-guard

Guards a real, reproducible correctness bug in recent PyTorch builds:
transferring an MPS (Apple Silicon GPU) tensor into a CPU `float64`
or `complex128` destination via `.to()` or `.copy_()` silently writes
**zeros** (or, for `.copy_()` into a pre-filled destination, leaves it
**completely unchanged**) instead of raising an error or converting
correctly. No exception, no warning — just wrong data.

Upstream issue: [pytorch/pytorch#197715](https://github.com/pytorch/pytorch/issues/197715)
("[MPS] Copy from MPS into a CPU float64 or complex128 tensor
silently writes nothing (2.14 regression)"), open as of this repo's
creation (2026-09-20), labeled `module: correctness (silent)` and
`module: mps`. A related PR, [#197722](https://github.com/pytorch/pytorch/pull/197722),
makes the failure *loud* (raises instead of silently no-oping) but
does not itself deliver a *correct converted result* — and is
unmerged as of this writing.

## Why this matters

Root cause (per the issue reporter, consistent with PyTorch's
`c10`/Metal indexing code): the Metal `store_at_offs` switch used by
MPS's copy/cast kernels has no case for `Double`/`ComplexDouble`
destinations. The store is a silent no-op for those two dtypes.

Any macOS/Apple-Silicon workflow that moves an MPS tensor to the CPU
for float64-precision post-processing (NumPy interop, scientific
reductions, complex-valued signal processing) or writes into a
pre-allocated float64/complex128 buffer via `copy_()` gets silently
wrong data with **no signal anything went wrong**.

## Independently reproduced on real hardware

This bug was reproduced from scratch on this repo's own build/test
host: **Apple M4, macOS 15.7.7, torch 2.14.0, a real (non-virtualized)
MPS device** — not simulated, not assumed from the issue text alone.

```
>>> x = torch.arange(1, 9, dtype=torch.float32, device="mps") / 3
>>> x.to("cpu", torch.float64)
tensor([0., 0., 0., 0., 0., 0., 0., 0.], dtype=torch.float64)   # WRONG: should be [0.3333, 0.6667, ...]

>>> dst = torch.full((8,), 7.0, dtype=torch.float64)
>>> dst.copy_(x)
>>> dst
tensor([7., 7., 7., 7., 7., 7., 7., 7.], dtype=torch.float64)   # WRONG: dst is completely unchanged
```

## The guard

`safe_to()` and `safe_copy_()` are drop-in-shaped replacements for
`.to()` / `.copy_()` that sidestep the defect entirely: for the two
affected dtype/MPS-source combinations, they transfer at the
**source's own dtype** first (which Metal always handles correctly)
and only upcast to `float64`/`complex128` as a second, CPU-side step.
Every other combination delegates straight to the native op
unchanged — no double-guarding of an already-correct path.

```python
from torch_mps_copy_dtype_guard import safe_to, safe_copy_

x = torch.arange(1, 9, dtype=torch.float32, device="mps") / 3
y = safe_to(x, "cpu", torch.float64)          # correct: [0.3333, 0.6667, ...]

dst = torch.full((8,), 7.0, dtype=torch.float64)
safe_copy_(dst, x)                             # correct: dst now holds the converted values
```

## CLI

```
$ pip install -e '.[torch]'
$ torch-mps-copy-dtype-guard
  torch version                9.9.9
  mps functional on this host  yes
● MPS-to-CPU float64/complex128 silent copy bug reproduced on this host
● guard matches the CPU-oracle conversion on every exercised device

per-dtype results (native .to()/.copy_() vs guarded vs CPU oracle)
  float64      native.to=SILENT-WRONG  native.copy_=SILENT-WRONG  guard.to=guard-ok  guard.copy_=guard-ok
  complex128   native.to=SILENT-WRONG  native.copy_=SILENT-WRONG  guard.to=guard-ok  guard.copy_=guard-ok
```

`--json` for machine-readable output, `--no-color` to disable ANSI
color (also respects `NO_COLOR`/`FORCE_COLOR`).

## Honest limitations

- **Only guards `.to()`/`.copy_()` as used through this module's
  functions.** It does not monkeypatch `torch.Tensor.to`/`copy_`
  globally — call `safe_to`/`safe_copy_` explicitly at the call sites
  that matter, or wherever your code moves MPS tensors to CPU
  float64/complex128 destinations.
- **MPS-specific.** On a host without a real, functional MPS device
  (including virtualized CI runners that report
  `torch.backends.mps.is_available() == True` but cannot actually
  allocate — see `actions/runner-images#9918`), the MPS-side tests
  are honestly skipped, not silently passed. This repo's own CI
  cannot exercise the MPS path (GitHub Actions has no real Apple
  Silicon GPU runners); the MPS reproduction and guard verification
  above were run on real Apple Silicon hardware, not CI.
  CPU-only usage of `safe_to`/`safe_copy_` is portable and tested on
  both `ubuntu-latest` and `macos-latest` in CI.
- **Tracks a specific upstream regression.** If PyTorch fixes
  `pytorch/pytorch#197715` upstream, the "native bug reproduced" test
  will start failing with a clear message on a host with functional
  MPS — that's a signal to re-check upstream status, not a defect in
  this guard. The guard functions remain safe to call either way
  (correct output regardless of whether the underlying bug is still
  present).

## Install

```
pip install -e '.[dev,torch]'   # for development/testing
pip install -e '.[torch]'       # for use
```

## License

MIT
