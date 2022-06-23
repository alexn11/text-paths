import torch
from bert_utils import BertModules, core_bert_loss_function

bert_modules = BertModules()
core_model = bert_modules.get_core_model()

print('moving model to GPU')
core_model.to('cuda')

layers = []
for i in range(11):
  layer = core_model.layer[i]
  layers += [
    layer.attention.self.query,
    layer.attention.self.key,
    layer.attention.self.value,
    layer.attention.output.dense,
    layer.attention.output.LayerNorm,
    layer.intermediate.dense,
    layer.output.dense,
    layer.output.LayerNorm,
  ]

print('checking layers')
for layer in layers:
  for p in layer.parameters():
    assert(p.device == torch.device('cuda:0'))

print('all done, all passed')

x = torch.randn(size=(2, 768))
x = x.to('cuda')

v = torch.nn.parameter.Parameter(x)
optimizer = torch.optim.Adam([ v ], lr=0.001)

optimizer.zero_grad()
y = core_model(v)



