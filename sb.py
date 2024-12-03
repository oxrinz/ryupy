import ryupy as rp

bank = rp.nn.LayerBank()
bank.layer1 = rp.nn.Linear(2, 5, rp.nn.InitType.KAIMING_NORMAL)
bank.layer2 = rp.nn.Linear(5, 3, rp.nn.InitType.KAIMING_NORMAL)


def forward(inputs):
    x = bank.layer1(inputs["sex"])
    print(x)
    x = rp.nn.relu(x)
    print(x)
    x = bank.layer2(x)
    print(x)
    x = rp.nn.relu(x)
    print(x)
    return {"sex": x}


input = {"sex": rp.fill([2], 2.0)}
target = rp.fill([3], 111.0)

loss = [1111]
optim = rp.nn.optim.SGD(bank)

steps_taken = 0

for i in range(111):
    out = forward(input)

    loss = rp.nn.loss.mse(out["sex"], target)

    loss.backward()

    optim.step()

    steps_taken += 1
