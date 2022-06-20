# systemd-run --user --pipe -p MemoryMax=6000M bash
# cd projects/text-paths
# python

import matplotlib.pyplot as pyplot
import torch
from torch.utils.tensorboard import SummaryWriter


torch_device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'Using {torch_device} device')

tensorboard_writer = SummaryWriter()

from input_optimizer import ModelInverter
from bert_utils import BertModules, core_bert_loss_function

start_sentence = 'There is no place like home.'
end_sentence_one_word = 'There is no place like Italy.'
end_sentence_keep_meaning = 'No place is like home.'
end_sentence_different = 'The cat jumps over the fence.'
nb_interpolation_steps = 10
# DEBUG:
nb_interpolation_steps = 2
# <<<<<
core_small_loss = 0.1
#core_n_max_steps = 1220
core_n_max_steps = 12 # 122
embedding_small_loss = 10
embedding_n_max_steps = 12 # 212



bert_modules = BertModules()
core_model = bert_modules.get_core_model()
embedding_model = bert_modules.get_embedding_model()

# 1st path:
start_text = start_sentence
end_text = end_sentence_one_word

start_token_seq, start_emb_input, start_emb_output, start_core_output = bert_modules.compute_text_steps(start_text)
start_core_input = start_emb_output

end_core_output = bert_modules.compute_core_model_output(end_text)

t = torch.linspace(0, 1, nb_interpolation_steps)

outputs = [ start_core_output*(1 - t[i]) + end_core_output*t[i] for i in range(nb_interpolation_steps) ]
embedding_vectors = [ start_emb_output ]


# core path

solver = ModelInverter(core_model,
                       start_core_input.detach().clone(),
                       loss_function=core_bert_loss_function,
                       tensorboard_writer=tensorboard_writer)

# >
for i in range(nb_interpolation_steps):
  # DEBUG:
  print(f'embed vect loop: {i}')
  # <<<
  #i = 0
  #if(True):
  solver.compute_inverse(outputs[i],
                         n_max_steps=core_n_max_steps,
                         min_loss=core_small_loss,
                         torch_device='cpu',
                         lr=0.01)#lr=0.0001) # DEBUG: 0.01<ok, 1.<bad, 0.1<bad, 0.02 < ok?
  embedding_vectors.append(solver.get_computed_solution())

# DEBUG:
print('computed embedding vectors:')
for i, v in enumerate(embedding_vectors):
  print(f'{i}: {v}')
# <<<<

# embedding path

input_vectors = [ start_emb_input ]

solver = ModelInverter(bert_modules.get_embedding_model(), start_emb_input.clone().detach())

# >
for i in range(nb_interpolation_steps):
  # DEBUG:
  print(f'input vect loop: {i}')
  # <<<
  #i = 0
  #if(True):
  solver.compute_inverse(embedding_vectors[i],
                         n_max_steps=embedding_n_max_steps,
                         min_loss=embedding_small_loss,
                         torch_device='cpu',
                         lr=0.01)
  input_vectors.append(solver.get_computed_solution())

# DEBUG:
print('computed input vectors:')
for i, v in enumerate(input_vectors):
  print(f'{i}: {v}')
# <<<<

# text path
# TODO translate the sequence of 1hot vectors into text (check how to use the Tokenizer)

def translate_inputs_to_text(tokenizer, input_vector):
  input_vector = input_vector[0]
  sequence = []
  for i in range(512):
    token = input_vector[i, :].argmax().item()
    if(abs(input_vector[i, token]) > 0.1):
      sequence.append(token)
    else:
      sequence.append(0)
  # TODO: not sure this works like that
  text = tokenizer.decode(sequence)
  return text

tokenizer = bert_modules.tokenizer
texts = []
for input_vector in input_vectors:
  text = translate_inputs_to_text(tokenizer, input_vector)
  texts.append(text)

# TODO: print the texts
for i, text in enumerate(texts):
  print(f'{i}: {text}')






























