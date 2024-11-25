import ryupy as rp

x = rp.rand([2, 2], grad=True)
w = rp.rand([2, 2], grad=True)

out = x + w

out.backward()

print(w)
print(w.grad)