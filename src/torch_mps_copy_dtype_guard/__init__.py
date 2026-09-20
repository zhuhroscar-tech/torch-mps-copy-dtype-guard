"""torch-mps-copy-dtype-guard: guard for a real MPS-to-CPU copy/cast
bug where float64 and complex128 CPU destinations are silently left
as zeros (or unchanged stale memory) when the source is an MPS
tensor (pytorch/pytorch#197715).
"""
__version__ = "0.1.0"
