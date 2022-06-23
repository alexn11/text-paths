# systemd-run --user --pipe -p MemoryMax=6000M bash
# cd projects/text-paths
# python
import argparse

import tqdm

import torch
from torch.utils.tensorboard import SummaryWriter

default_start_sentence = 'I like pizzas.'
end_sentence_one_word = 'I hate pizzas.'
end_sentence_keep_meaning = 'I like any pizzas.'
end_sentence_different = 'It is raining outside.'
default_end_sentence = end_sentence_keep_meaning


# python bert_text_path.py --device cpu --core-n-max-steps 6214 --nb-interpolation-steps 42
# python bert_text_path.py --device cpu --weight-decay 0.1 --core-n-max-steps 614 --nb-interpolation-steps 10

argparser = argparse.ArgumentParser()
argparser.add_argument('--lr', type=float, default=0.01)
argparser.add_argument('--device', type=str, default='auto')
argparser.add_argument('--core-n-max-steps', type=int, default=332)
argparser.add_argument('--core-small-loss', type=float, default=13500)#11080)
argparser.add_argument('--nb-interpolation-steps', type=int, default=21)
argparser.add_argument('--start-sentence', type=str, default=default_start_sentence)
argparser.add_argument('--end-sentence', type=str, default=default_end_sentence)
argparser.add_argument('--weight-decay', type=float, default=1.e-5)
argparser.add_argument('--embedding-small-loss', type=int, default=10)
argparser.add_argument('--embedding-n-max-steps', type=int, default=212)# = 12

parsed_arguments = argparser.parse_args()

print('command line arguments values:')
print(parsed_arguments)

start_sentence = parsed_arguments.start_sentence
end_sentence = parsed_arguments.end_sentence

torch_device = parsed_arguments.device

nb_interpolation_steps = parsed_arguments.nb_interpolation_steps
core_n_max_steps = parsed_arguments.core_n_max_steps
core_small_loss = parsed_arguments.core_small_loss
embedding_n_max_steps = parsed_arguments.embedding_n_max_steps
embedding_small_loss = parsed_arguments.embedding_small_loss
learning_rate = parsed_arguments.lr
weight_decay = parsed_arguments.weight_decay

if(torch_device == 'auto'):
  torch_device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'Using device: {torch_device}')

tensorboard_writer = SummaryWriter()

from input_optimizer import ModelInverter
from bert_utils import BertModules, core_bert_loss_function

bert_modules = BertModules()
core_model = bert_modules.get_core_model()
embedding_model = bert_modules.get_embedding_model()

start_text = start_sentence
end_text = end_sentence

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
                       torch_device=torch_device,
                       tensorboard_writer=tensorboard_writer)
print('pass: core')
interpolation_progress_bar = tqdm.tqdm(range(nb_interpolation_steps))
for i in interpolation_progress_bar:
  loss_history = solver.compute_inverse(outputs[i],
                         n_max_steps=core_n_max_steps,
                         min_loss=core_small_loss,
                         lr=learning_rate,
                         optimizer_kwargs={
                          'weight_decay': weight_decay,
                         })
  embedding_vectors.append(solver.get_computed_solution())
  interpolation_progress_bar.set_postfix({ 'loss': loss_history[-1] })


# embedding path

input_vectors = [ start_emb_input ]

solver = ModelInverter(bert_modules.get_embedding_model(),
                       start_emb_input.clone().detach(),
                       torch_device=torch_device)

print('pass: embedding')
interpolation_progress_bar = tqdm.tqdm(range(nb_interpolation_steps))
for i in interpolation_progress_bar:
  loss_history = solver.compute_inverse(embedding_vectors[i],
                                        n_max_steps=embedding_n_max_steps,
                                        min_loss=embedding_small_loss,
                                        lr=0.01)
  input_vectors.append(solver.get_computed_solution())
  interpolation_progress_bar.set_postfix({ 'loss': loss_history[-1] })


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






























