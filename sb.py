import ryupy as rp

layer = rp.cuda.nn.Linear(10, 10, rp.cuda.nn.InitType.XAVIER_UNIFORM)

tensor1 = rp.cuda.rand([10,10], -10, 10)

tensor2 = rp.cuda.rand([10,10], -10, 10)

ten_out = tensor1 @ tensor2


l1_out = layer.forward(tensor1)

print(l1_out)