import ryupy as rp

x = rp.rand([2, 2], grad=True)
w1 = rp.randn([2, 2], grad=True)
w2 = rp.randn([2, 2], grad=True)

z1 = x * w1
out = z1 * w2

out.backward()

print(w1.grad)
print(x * w2)