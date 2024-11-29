import ryupy as rp

bank = rp.nn.LayerBank()
bank.layer1 = rp.nn.Linear(2, 2, rp.nn.InitType.KAIMING_NORMAL)


def forward(x):
    return bank.layer1(x)


model = rp.nn.Net(bank, forward)

input = rp.fill([2], 2.0)
target = rp.rand([2])

out = model(input)

loss = [1111]

optim = rp.nn.optim.SGD(model)

while loss[0] > 0.01:
    loss = rp.nn.loss.mse(out, target)
    loss.backward()

    optim.step()

    out = model(input)

    print(out)
    print(f"loss: {loss[0]}")