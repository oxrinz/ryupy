from __future__ import annotations
import ryupy.cuda.nn
import typing
import ryupy.cuda

__all__ = [
    "InitType",
    "KAIMING_NORMAL",
    "KAIMING_UNIFORM",
    "Linear",
    "XAVIER_NORMAL",
    "XAVIER_UNIFORM"
]


class InitType():
    """
    Members:

      XAVIER_UNIFORM

      XAVIER_NORMAL

      KAIMING_UNIFORM

      KAIMING_NORMAL
    """
    def __eq__(self, other: object) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: int) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: object) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, state: int) -> None: ...
    def __str__(self) -> str: ...
    @property
    def name(self) -> str:
        """
        :type: str
        """
    @property
    def value(self) -> int:
        """
        :type: int
        """
    KAIMING_NORMAL: ryupy.cuda.nn.InitType # value = <InitType.KAIMING_NORMAL: 3>
    KAIMING_UNIFORM: ryupy.cuda.nn.InitType # value = <InitType.KAIMING_UNIFORM: 2>
    XAVIER_NORMAL: ryupy.cuda.nn.InitType # value = <InitType.XAVIER_NORMAL: 1>
    XAVIER_UNIFORM: ryupy.cuda.nn.InitType # value = <InitType.XAVIER_UNIFORM: 0>
    __members__: dict # value = {'XAVIER_UNIFORM': <InitType.XAVIER_UNIFORM: 0>, 'XAVIER_NORMAL': <InitType.XAVIER_NORMAL: 1>, 'KAIMING_UNIFORM': <InitType.KAIMING_UNIFORM: 2>, 'KAIMING_NORMAL': <InitType.KAIMING_NORMAL: 3>}
    pass
class Linear():
    def __init__(self, arg0: int, arg1: int, arg2: InitType) -> None: ...
    def forward(self, arg0: ryupy.cuda.Tensor) -> ryupy.cuda.Tensor: ...
    @property
    def bias(self) -> ryupy.cuda.Tensor:
        """
        :type: ryupy.cuda.Tensor
        """
    @bias.setter
    def bias(self, arg0: ryupy.cuda.Tensor) -> None:
        pass
    @property
    def weight(self) -> ryupy.cuda.Tensor:
        """
        :type: ryupy.cuda.Tensor
        """
    @weight.setter
    def weight(self, arg0: ryupy.cuda.Tensor) -> None:
        pass
    pass
KAIMING_NORMAL: ryupy.cuda.nn.InitType # value = <InitType.KAIMING_NORMAL: 3>
KAIMING_UNIFORM: ryupy.cuda.nn.InitType # value = <InitType.KAIMING_UNIFORM: 2>
XAVIER_NORMAL: ryupy.cuda.nn.InitType # value = <InitType.XAVIER_NORMAL: 1>
XAVIER_UNIFORM: ryupy.cuda.nn.InitType # value = <InitType.XAVIER_UNIFORM: 0>
