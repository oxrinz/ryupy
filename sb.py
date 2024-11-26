import ryupy as rp

x = rp.randn([2, 2])
w1 = rp.randn([2, 2], grad=True)
w2 = rp.randn([2, 2])
w3 = rp.randn([2, 2])

z1 = x @ w1
z2 = z1 @ w2
out = z2 @ w3

out.backward()

w1_grad_expected = (rp.ones([2, 2]) @ w3) @ w2 @ x

print("Actual w1.grad:")
print(w1.grad)

print("Expected w1.grad:")
print(w1_grad_expected)
