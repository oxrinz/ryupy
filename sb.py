import sys, os

sys.path.append(os.path.abspath("src/py"))

import ryupy as rpy

t1 = rpy.Tensor([3, 0.5, 6])
t2 = rpy.Tensor([1, 7, 2])

t1 *= t2

print(t1)
