import ryupy as rp

bank = rp.nn.LayerBank()
bank.layer1 = rp.nn.Linear(2, 5, rp.nn.InitType.KAIMING_NORMAL)
bank.layer2 = rp.nn.Linear(5, 3, rp.nn.InitType.KAIMING_NORMAL)


def forward(x):
    x = bank.layer1(x)
    return bank.layer2(x)


model = rp.nn.Net(bank, forward)

input = rp.fill([2], 2.0)
target = rp.rand([3])

out = model(input)

loss = [1111]

optim = rp.nn.optim.SGD(model)

while loss[0] > 0.0001:
    loss = rp.nn.loss.mse(out, target)
    loss.backward()

    optim.step()

    out = model(input)

    print(out)
    print(f"loss: {loss[0]}")