from __future__ import annotations
import ryupy
import typing
__all__ = ['Tensor']
class Tensor(ryupy._Tensor):
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self, arg0: typing.Any) -> None:
        ...
    @property
    def data(self) -> typing.Any:
        ...
    @property
    def flattenedData(self) -> typing.Any:
        ...
    @property
    def shape(self) -> list[int]:
        ...
