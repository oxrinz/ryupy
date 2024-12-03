import ryupy as rp

bank = rp.nn.LayerBank()
bank.layer1 = rp.nn.Linear(2, 5, rp.nn.InitType.KAIMING_NORMAL)
bank.layer2 = rp.nn.Linear(5, 3, rp.nn.InitType.KAIMING_NORMAL)


def forward(inputs):
    x = bank.layer1(inputs["sex"])
    print(x)
    return {"sex": bank.layer2(x)}


input = {"sex": rp.fill([2], 2.0)}
target = rp.fill([3], 11.0)

loss = [1111]
optim = rp.nn.optim.SGD(bank)

steps_taken = 0

dataset = rp.Dataset(inputs={"sex": [rp.rand([3])]}, targets={"sex": [rp.rand([3])]})

while loss[0] > 0.0001:
    out = forward(input)

    loss = rp.nn.loss.mse(out["sex"], target)
    loss.backward()

    optim.step()

    steps_taken += 1

    print(out["sex"])
    print(f"loss: {loss[0]}")

print(bank.layer1.bias)
print(bank.layer1.weight)
print(steps_taken)
