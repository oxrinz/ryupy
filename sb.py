import ryupy as rp

tensor = rp.rand([3, 3])

print(tensor)
tensor[2][1] = 1111
tensor2 = tensor[2][2]
print(tensor2)
