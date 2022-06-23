# systemd-run --user --pipe -p MemoryMax=6000M bash
# cd projects/text-paths
# python
import argparse
import re
import sys

import matplotlib.pyplot as pyplot
import torch
from torch.utils.tensorboard import SummaryWriter


argparser = argparse.ArgumentParser()
argparser.add_argument('--lr', type=float, default=0.01)
argparser.add_argument('--device', type=str, default='auto')
argparser.add_argument('--core-n-max-steps', type=int, default=332)
argparser.add_argument('--core-small-loss', type=float, default=13500)#11080)
argparser.add_argument('--nb-interpolation-steps', type=int, default=21)
parsed_arguments = argparser.parse_args()

print('command line arguments values:')
print(parsed_arguments)

learning_rate = parsed_arguments.lr
torch_device = parsed_arguments.device
core_n_max_steps = parsed_arguments.core_n_max_steps
core_small_loss = parsed_arguments.core_small_loss
nb_interpolation_steps = parsed_arguments.nb_interpolation_steps

if(torch_device == 'auto'):
  torch_device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'Using device: {torch_device}')

tensorboard_writer = SummaryWriter()

from input_optimizer import ModelInverter
from bert_utils import BertModules, core_bert_loss_function

start_sentence = 'There is no place like home.'
end_sentence_one_word = 'There is no place like Italy.'
end_sentence_keep_meaning = 'No place is like home.'
end_sentence_different = 'The cat jumps over the fence.'

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

for i in range(nb_interpolation_steps):
  solver.compute_inverse(outputs[i],
                         n_max_steps=core_n_max_steps,
                         min_loss=core_small_loss,
                         torch_device=torch_device,#'cpu',
                         lr=learning_rate)#lr=0.01)#lr=0.0001) # DEBUG: 0.01<ok, 1.<bad, 0.1<bad, 0.02 < ok?
  embedding_vectors.append(solver.get_computed_solution())


# embedding path

input_vectors = [ start_emb_input ]

solver = ModelInverter(bert_modules.get_embedding_model(), start_emb_input.clone().detach())

for i in range(nb_interpolation_steps):
  solver.compute_inverse(embedding_vectors[i],
                         n_max_steps=embedding_n_max_steps,
                         min_loss=embedding_small_loss,
                         torch_device=torch_device, #'cpu',
                         lr=0.01)
  input_vectors.append(solver.get_computed_solution())


# text path

def remove_special_tokens(text):
  text = text.split('.')[0]
  # could probably use regex instead
  text = text.replace('[PAD] ', '')
  text = text.replace('[PAD]', '')
  text = text.replace('[CLS] ', '')
  text = text.replace('[SEP] ', '')
  text = text.replace('[MASK] ', '')
  text = text.replace('[MASK]', '')
  return text

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
  text = remove_special_tokens(text)
  return text

tokenizer = bert_modules.tokenizer
texts = []
for input_vector in input_vectors:
  text = translate_inputs_to_text(tokenizer, input_vector)
  texts.append(text)

for i, text in enumerate(texts):
  print(f'{i}: {text}')






























