from __future__ import annotations
from . import cpu
from . import cuda
__all__ = ['cpu', 'cuda']
class _ITensor:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
