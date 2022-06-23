
import torch
from torch import nn
import matplotlib.pyplot as pyplot



torch_device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'Using {torch_device} device')


#model_depth = 1

model_dimension = 5
nb_input_attempts = 18
perturbation_size = 0.1


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(model_dimension, model_dimension),
            nn.ReLU(),
            #nn.Linear(model_dimension, model_dimension),
        )
    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits



model = NeuralNetwork().to(torch_device)
model.eval()

#print(model)


input_vectors = torch.rand((nb_input_attempts, model_dimension),
                           dtype=torch.float,
                           device=torch_device,
                           requires_grad=True)

expected_output_vector = model(torch.rand(model_dimension,
                                          dtype=torch.float,
                                          device=torch_device,
                                          requires_grad=False)).detach()
expected_output_vector = expected_output_vector * (1.+perturbation_size * torch.rand(model_dimension, device=torch_device))
expected_output = expected_output_vector.repeat((nb_input_attempts, 1))

# new version
from input_optimizer import ModelInverter
from torch.utils.tensorboard import SummaryWriter

tensorboard_writer = SummaryWriter()

model_inverter = ModelInverter(model,
                               input_vectors,
                               loss_function=nn.MSELoss(reduction='sum'),
                               tensorboard_writer=tensorboard_writer)
loss_history = model_inverter.compute_inverse(expected_output,
                                              n_max_steps=1000,
                                              torch_device=torch_device,
                                              lr=0.1)




output = model(model_inverter.get_computed_solution())
output.mean(0)
expected_output_vector

# (sort of work, not particularly good)

# -----------------------------------------------------------------
# old version:
class InputOptimizer(nn.Module):
  def __init__(self, model, input_vectors):
    super(InputOptimizer, self).__init__()
    self.model = model
    self.input_vectors = nn.parameter.Parameter(input_vectors)
    self.register_parameter('input_vector', self.input_vectors)
  def forward(self):
    return self.model(self.input_vectors)

input_optimizer = InputOptimizer(model, input_vectors).to(torch_device)

print(input_optimizer)

loss_function = nn.MSELoss(reduction='sum')
#loss_function = nn.L1Loss(reduction='sum')

parameters_to_optimize = [ input_optimizer.input_vectors ]
#optimizer = torch.optim.SGD(parameters_to_optimize, lr=0.1)
optimizer = torch.optim.Adam(parameters_to_optimize, lr=0.0001)

#print('before')
#for p in input_optimizer.parameters():
#  print(p)
#
#print(input_optimizer.input_vector)


input_optimizer.train()

optimizer.zero_grad()
loss_function(input_optimizer(), expected_output).backward()
optimizer.step()

#print('after')
#for p in input_optimizer.parameters():
#  print(p)
#
#print(input_optimizer.input_vector)



loss_values = []

for i in range(10000):
  optimizer.zero_grad()
  loss = loss_function(input_optimizer(), expected_output)
  loss.backward()
  loss_values.append(loss.item())
  optimizer.step()


pyplot.plot(loss_values)
pyplot.show()




output = model(parameters_to_optimize[0])
output.mean(0)
expected_output_vector
# -----------------------------------------------------------------



#  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> INCEPTIONv3
# new ver
import numpy
import torch
from torch import nn
from input_optimizer import ModelInverter
from torch.utils.tensorboard import SummaryWriter

torch_device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'Using {torch_device} device')

tensorboard_writer = SummaryWriter()

model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True).to(torch_device)

norm_mean = [ 0.485, 0.456, 0.406 ]
norm_std = [ 0.229, 0.224, 0.225 ]

norm_mean = torch.tensor(norm_mean).repeat(299, 299, 1).T # < deprecation warning about T
norm_std = torch.tensor(norm_std).repeat(299, 299, 1).T

norm_mean = norm_mean.to(torch_device)
norm_std = norm_std.to(torch_device)

input_vector = (torch.rand((1, 3, 299, 299),
                           dtype=torch.float,
                           device=torch_device,
                           requires_grad=True) - norm_mean) / norm_std

#input_vector.to(torch_device)

model.eval()
nothing = model(input_vector)
# input shape: batch_size x 3 x 299 x 299
# output shape: batch_size x 1000

expected_output_vector = numpy.zeros(shape=nothing.shape)
expected_output_vector[0, 31] = 1.
expected_output_vector = torch.tensor(expected_output_vector,
                                      dtype=torch.float,
                                      device=torch_device)


model_inverter = ModelInverter(model,
                               input_vector,
                               loss_function=nn.MSELoss(reduction='sum'),
                               tensorboard_writer=tensorboard_writer)
loss_history = model_inverter.compute_inverse(expected_output_vector,
                                              n_max_steps=10000,
                                              torch_device=torch_device,
                                              lr=0.0001)




output = model(model_inverter.get_computed_solution())
output.mean(0)
expected_output_vector



# -----------------------------------------------------------------
# old ver
import torch
from torch import nn
import matplotlib.pyplot as pyplot
import numpy


torch_device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'Using {torch_device} device')



class InputOptimizer(nn.Module):
  def __init__(self, model, input_vectors):
    super(InputOptimizer, self).__init__()
    self.model = model
    self.input_vectors = nn.parameter.Parameter(input_vectors)
    self.register_parameter('input_vector', self.input_vectors)
  def forward(self):
    return self.model(self.input_vectors)

model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True).to(torch_device)

norm_mean = [ 0.485, 0.456, 0.406 ]
norm_std = [ 0.229, 0.224, 0.225 ]

norm_mean = torch.tensor(norm_mean).repeat(299, 299, 1).T
norm_std = torch.tensor(norm_std).repeat(299, 299, 1).T

input_vector = (torch.rand((3, 299, 299), dtype=torch.float, device=torch_device, requires_grad=True) - norm_mean) / norm_std

input_vector.to(torch_device)

model.eval()
nothing = model(input_vector)
# 3.299.299

expected_output_vector = numpy.zeros(1000)
expected_output_vector[31] = 1.
expected_output_vector = torch.tensor(expected_output_vector)

input_optimizer = InputOptimizer(model, input_vector).to(torch_device)


loss_function = nn.MSELoss(reduction='sum')


parameters_to_optimize = [ input_optimizer.input_vectors ]
optimizer = torch.optim.Adam(parameters_to_optimize, lr=0.0001)

input_optimizer.train()

optimizer.zero_grad()
loss_function(input_optimizer(), expected_output).backward()
optimizer.step()


loss_values = []

for i in range(10000):
  optimizer.zero_grad()
  loss = loss_function(input_optimizer(), expected_output)
  loss.backward()
  loss_values.append(loss.item())
  optimizer.step()


pyplot.plot(loss_values)
pyplot.show()




























>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> BERT

"""
ideas:
- try with an actual output -> seems to work
- try with a small perturbation of an actual output -> also works!
"""

import torch
from torch import nn
import matplotlib.pyplot as pyplot



torch_device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'Using {torch_device} device')


"""
class InputOptimizerBERT(nn.Module):
  def __init__(self, model, input_vectors):
    super(InputOptimizerBERT, self).__init__()
    self.model = model
    self.input_vectors = nn.parameter.Parameter(input_vectors)
    self.register_parameter('input_vector', self.input_vectors)
  def forward(self):
    return self.model(self.input_vectors, retain_graph=True)[0]
"""

class InputOptimizer(nn.Module):
  def __init__(self, model, input_vectors):
    super(InputOptimizer, self).__init__()
    self.model = model
    self.input_vectors = nn.parameter.Parameter(input_vectors)
    self.register_parameter('input_vector', self.input_vectors)
  def forward(self):
    return self.model(self.input_vectors)


def bert_optimizer_loss(output, target):
  return loss_function(output.last_hidden_state, target.last_hidden_state)

# i want the bare model that output vector representations of sequences
from transformers import BertModel
bert_model_name = 'bert-base-uncased'

# load model into torch

bert_model = BertModel.from_pretrained(bert_model_name)

bert_model = bert_model.to(torch_device)

# need the tokenizer etc. to prepare the inputs to the model
from transformers import BertTokenizer

bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
model_input_length = bert_tokenizer.max_model_input_sizes[bert_model_name]
max_token = len(bert_tokenizer.vocab)

"""
text = 'this is a test.'
v = bert_tokenizer.encode(text,
                      return_tensors='pt',
                      max_length=model_input_length,
                      pad_to_max_length=True)

y = bert_model(v)
"""

def encode_text_as_input(text):
  return bert_tokenizer.encode(text,
                      return_tensors='pt',
                      max_length=model_input_length,
                      pad_to_max_length=True)


start_sentence = 'There is no place like home.'
end_sentence_one_word = 'There is no place like Italy.'
end_sentence_keep_meaning = 'No place is like home.'
end_sentence_different = 'The cat jumps over the fence.'


embedding = bert_model.embeddings
core_model = bert_model.encoder

start_input = embedding(encode_text_as_input(start_sentence))
expected_output = core_model(start_input)

embedded_shape = start_input.shape

perturbation_size = 0.1
#    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> HERE!  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> HERE!  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> HERE!  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> HERE!  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> HERE!  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#################################
input_parameter = start_input * (1. + perturbation_size * torch.rand(start_input.shape))

input_optimizer = InputOptimizer(core_model, input_parameters).to(torch_device)

loss_function = nn.MSELoss(reduction='sum')
parameters_to_optimize = [ input_optimizer.input_vectors ]
optimizer = torch.optim.Adam(parameters_to_optimize, lr=0.0001)


# because of recursion, need to detach_() hidden recursive states
# ideas
# which are the recursive bits in bert? -> detach them -> there is none ofc, so WHAT IS THE PROBLEM??????
#OR
#use retain_grad thing (come back to docs)


input_optimizer.train()

"""
############### test step
optimizer.zero_grad()
current_output = input_optimizer()
bert_optimizer_loss(current_output, expected_output).backward(retain_graph=True)
optimizer.step()
#current_output.last_hidden_state.detach_()

##########################
"""

loss_values = []

for i in range(14):
  optimizer.zero_grad()
  current_output = input_optimizer()
  loss = bert_optimizer_loss(current_output, expected_output)
  loss.backward(retain_graph=True)
  #current_output.last_hidden_state.detach_()
  loss_values.append(loss.item())
  optimizer.step()


pyplot.plot(loss_values)
pyplot.show()

"""
step 1:
- optimize the internal representation of the input
step 2:
- optmize for the embedding on 1-hot vectors
step 3:
- translate vector in 1-hot space to tokens (max or randomly according to Boltz. distrib)
"""

"""
y.last_hidden_state or y[0]
y.pooler_output or y[1]

I am always lying.

will need to reverse the embedding

"""


"""

ideas:
- use either the last hidden state OR the attention! OR the pooler output


FROM THE DOC
    >>> from transformers import BertTokenizer, BertModel
 |          >>> import torch
 |      
 |          >>> tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
 |          >>> model = BertModel.from_pretrained('bert-base-uncased')
 |      
 |          >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
 |          >>> outputs = model(**inputs)
 |      
 |          >>> last_hidden_states = outputs.last_hidden_state



  (embeddings): BertEmbeddings(
    (word_embeddings): Embedding(30522, 768, padding_idx=0)
    (position_embeddings): Embedding(512, 768)
    (token_type_embeddings): Embedding(2, 768)
    (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
    (dropout): Dropout(p=0.1, inplace=False)
  )



"""


































