import ryupy as rp

bank = rp.nn.LayerBank()
bank.layer1 = rp.nn.Linear(3, 5, rp.nn.InitType.KAIMING_NORMAL)

def forward(x):
    x = bank.layer1(x)
    return x

model = rp.nn.Net(bank, forward)

tensor = rp.rand([1, 3])
out = model(tensor)
