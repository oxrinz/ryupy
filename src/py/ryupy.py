import os
import sys

import ryupycuda


class Tensor:
    def __init__(self, vec):
        self.tensor = ryupycuda.tensor.Tensor(vec)

    def __add__(self, other):
        return Tensor(ryupycuda.math.vec_add(self.data, other.data))

    def __mul__(self, other):
        return Tensor(ryupycuda.math.vec_mult(self.data, other.data))

    @property
    def data(self):
        return self.tensor.vector
    
    def __str__(self):
        return f"Tensor with data: {self.data}"
