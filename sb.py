import ryupy as rp

ten1 = rp.rand([2, 2], grad=True)
ten2 = rp.rand([2, 2])

out = ten1 @ ten2

target = rp.ones([2, 2])

print(out)
print(target)

loss = rp.nn.loss.mse(out, target)

print(loss)