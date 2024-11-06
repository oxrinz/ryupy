import ryupy as rp
import numpy as np

rp.cu

rptensor1 = rp.cuda.Tensor(
    [[[[2, 3], [4, 5]], [[2, 3], [4, 5]]], [[[2, 3], [4, 5]], [[2, 3], [4, 5]]]]
)
rptensor2 = rp.cuda.Tensor(
    [[[[2, 3], [4, 5]], [[2, 3], [4, 5]]], [[[2, 3], [4, 5]], [[2, 3], [4, 5]]]]
)

nptensor1 = np.array(
    [[[[2, 3], [4, 5]], [[2, 3], [4, 5]]], [[[2, 3], [4, 5]], [[2, 3], [4, 5]]]]
)
nptensor2 = np.array(
    [[[[2, 3], [4, 5]], [[2, 3], [4, 5]]], [[[2, 3], [4, 5]], [[2, 3], [4, 5]]]]
)

rptensor = rptensor1 @ rptensor2
nptensor = nptensor1 @ nptensor2

print(rptensor.shape)
print(rptensor.data)
print(nptensor)