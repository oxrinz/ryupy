import ryupy as rp

layer = rp.cuda.nn.Linear(10, 10, rp.cuda.nn.InitType.XAVIER_UNIFORM)

print(layer.weight.shape)