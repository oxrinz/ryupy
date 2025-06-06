from typing import List, Optional, Tuple, Union, Callable, overload
from enum import Enum
import numpy.typing as npt
import numpy as np

class Tensor:
    shape: List[int]
    flattenedData: npt.NDArray[np.float32]
    data: npt.NDArray[np.float32]
    grad: Optional["Tensor"]
    requires_grad: bool

    def backward(self, gradient: Optional["Tensor"] = None) -> None: ...
    def copy(self) -> "Tensor": ...
    def __repr__(self) -> str: ...
    def __getitem__(self, index: int) -> Union[float, "Tensor"]: ...
    def __setitem__(self, index: int, value: Union[float, "Tensor"]) -> None: ...
    def __add__(self, other: "Tensor") -> "Tensor": ...
    def __sub__(self, other: "Tensor") -> "Tensor": ...
    def __mul__(self, other: "Tensor") -> "Tensor": ...
    def __truediv__(self, other: "Tensor") -> "Tensor": ...
    def __mod__(self, other: "Tensor") -> "Tensor": ...
    def __iadd__(self, other: "Tensor") -> "Tensor": ...
    def __isub__(self, other: "Tensor") -> "Tensor": ...
    def __imul__(self, other: "Tensor") -> "Tensor": ...
    def __itruediv__(self, other: "Tensor") -> "Tensor": ...
    def __imod__(self, other: "Tensor") -> "Tensor": ...
    def __pow__(self, other: "Tensor") -> "Tensor": ...
    def __ipow__(self, other: "Tensor") -> "Tensor": ...
    def __eq__(self, other: "Tensor") -> "Tensor": ...  # type: ignore
    def __ne__(self, other: "Tensor") -> "Tensor": ...  # type: ignore
    def __lt__(self, other: "Tensor") -> "Tensor": ...
    def __le__(self, other: "Tensor") -> "Tensor": ...
    def __gt__(self, other: "Tensor") -> "Tensor": ...
    def __ge__(self, other: "Tensor") -> "Tensor": ...
    def __and__(self, other: "Tensor") -> "Tensor": ...
    def __or__(self, other: "Tensor") -> "Tensor": ...
    def __xor__(self, other: "Tensor") -> "Tensor": ...
    def __invert__(self) -> "Tensor": ...
    def __lshift__(self, shift: int) -> "Tensor": ...
    def __rshift__(self, shift: int) -> "Tensor": ...
    def __iand__(self, other: "Tensor") -> "Tensor": ...
    def __ior__(self, other: "Tensor") -> "Tensor": ...
    def __ixor__(self, other: "Tensor") -> "Tensor": ...
    def __ilshift__(self, shift: int) -> "Tensor": ...
    def __irshift__(self, shift: int) -> "Tensor": ...
    def __neg__(self) -> "Tensor": ...
    def __matmul__(self, other: "Tensor") -> "Tensor": ...

def zeros(shape: List[int], *, grad: bool = False) -> Tensor: ...
def ones(shape: List[int], *, grad: bool = False) -> Tensor: ...
def fill(shape: List[int], value: float, *, grad: bool = False) -> Tensor: ...
def arange(
    start: float, stop: float, *, step: float = 1.0, grad: bool = False
) -> Tensor: ...
def linspace(start: float, stop: float, num: int, *, grad: bool = False) -> Tensor: ...
def rand(
    shape: List[int], *, low: float = 0.0, high: float = 1.0, grad: bool = False
) -> Tensor: ...
def randn(
    shape: List[int], *, mean: float = 0.0, std: float = 1.0, grad: bool = False
) -> Tensor: ...
def ryu() -> None: ...

class nn:
    class InitType(Enum):
        XAVIER_UNIFORM: "nn.InitType"
        XAVIER_NORMAL: "nn.InitType"
        KAIMING_UNIFORM: "nn.InitType"
        KAIMING_NORMAL: "nn.InitType"

    class Layer:
        def forward(self, x: Tensor) -> Tensor: ...
        def __call__(self, x: Tensor) -> Tensor: ...

    class Linear(Layer):
        weight: Tensor
        bias: Tensor

        @staticmethod
        def create(in_features: int, out_features: int) -> "nn.Linear": ...

    class LayerBank:
        @staticmethod
        def create() -> "nn.LayerBank": ...
        def __setattr__(self, name: str, layer: Layer) -> None: ...
        def __getattr__(self, name: str) -> Layer: ...

    class Net:
        @staticmethod
        def create() -> "nn.Net": ...
        def __call__(self, x: Tensor) -> Tensor: ...

    class loss:
        @staticmethod
        def mse(pred: Tensor, target: Tensor) -> Tensor: ...
