import ryupy as rp

layer = rp.nn.Linear(2, 2, rp.nn.InitType.XAVIER_UNIFORM)

tensor1 = rp.rand([2, 2], -10, 10)

tensor2 = rp.rand([2, 2], -10, 10)

layer.weight = tensor2

ten_out = tensor1 @ tensor2

l1_out = layer.forward(tensor1)

print(ten_out.data)
print(l1_out)