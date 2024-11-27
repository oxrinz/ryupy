import ryupy as rp

bank = rp.nn.LayerBank()

bank.layer1 = rp.nn.Linear(5, 5, rp.nn.InitType.KAIMING_NORMAL)

def forward(x):
    x = bank.layer1(x)
    return x

model = rp.nn.net(bank, forward)

trainer = rp.nn.trainers.Burt(model)

dataset = rp.data.Dataset(rp.randn([5]), rp.ones([5]))

trainer.train(dataset) 
