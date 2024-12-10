import ryupy as rp

bank = rp.nn.LayerBank()
bank.layer1 = rp.nn.Linear(2, 5, rp.nn.InitType.KAIMING_NORMAL)
bank.layer2 = rp.nn.Linear(5, 3, rp.nn.InitType.KAIMING_NORMAL)

def forward(x):

    x = bank.layer1(x)

    x = bank.layer2(x)

    return x


optim = rp.nn.optim.SGD(bank)

input = rp.arange(0, 2)
target = rp.fill([3], 0)


loss = [11111]
optim = rp.nn.optim.SGD(bank)

first = True

while loss[0] > 0.01:    
    out = forward(input)
    
    loss = rp.nn.loss.mse(out, target)

    loss.backward()
    
    optim.step()
    
    bank.zero_grad()
    
    print(loss)