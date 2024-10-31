import ryupy

tensor = ryupy.cuda.Tensor(
    [[[625, 2524, 48], [625, 2524, 48]], [[625, 2524, 48], [625, 2524, 48]]]
)

print(tensor.data)
print(tensor.shape)
print(tensor.flattenedData)
