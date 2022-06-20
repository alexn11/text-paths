


import torch
from torch import nn
import matplotlib.pyplot as pyplot
from transformers import BertModel
from transformers import BertTokenizer


torch_device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'Using {torch_device} device')




"""
(embeddings): BertEmbeddings(
    (word_embeddings): Embedding(30522, 768, padding_idx=0)
    (position_embeddings): Embedding(512, 768)
    (token_type_embeddings): Embedding(2, 768)
    (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
    (dropout): Dropout(p=0.1, inplace=False)
  )
"""

class InputOptimizer(nn.Module):
  def __init__(self, model, input_vectors):
    super(InputOptimizer, self).__init__()
    self.model = model
    self.input_vectors = nn.parameter.Parameter(input_vectors)
    self.register_parameter('input_vector', self.input_vectors)
    #print('parameters after register:')
    #for i, p in enumerate(self.parameters()):
    #  print(f'  {i}: {p.shape}')
    #  print(p)
    #  raise Exception('stop')
  def forward(self):
    return self.model(self.input_vectors)

class BertEmbeddings1Hot(nn.Module):
  def __init__(self,
               embedding_matrix,
               token_type_vector,
               position_embedding_vector,
               layer_normalization_module):
    super(BertEmbeddings1Hot, self).__init__()
    self.embedding_matrix = embedding_matrix
    self.token_type_vector = token_type_vector
    self.position_embedding_vector = position_embedding_vector
    self.layer_normalization = layer_normalization_module
  def forward(self, x):
    y = torch.matmul(x, self.embedding_matrix)
    y = y + self.position_embedding_vector + self.token_type_vector
    y = embedding_layer_norm(y)
    return y



bert_model_name = 'bert-base-uncased'
bert_model = BertModel.from_pretrained(bert_model_name)
bert_model = bert_model.to(torch_device)

bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
model_input_length = bert_tokenizer.max_model_input_sizes[bert_model_name]
max_token = len(bert_tokenizer.vocab)


def encode_text_as_input(text):
  return bert_tokenizer.encode(text,
                      return_tensors='pt',
                      max_length=model_input_length,
                      pad_to_max_length=True)




embedding = bert_model.embeddings
word_embedding = embedding.word_embeddings
position_embedding = embedding.position_embeddings
token_type_embedding = embedding.token_type_embeddings
embedding_layer_norm = embedding.LayerNorm


embedding_matrix = word_embedding.state_dict()['weight']
position_embedding_vector = position_embedding.state_dict()['weight']
token_type_vector = token_type_embedding.state_dict()['weight'][0,:]


bert_embedding_model = BertEmbeddings1Hot(embedding_matrix,
                                          token_type_vector,
                                          position_embedding_vector,
                                          embedding_layer_norm)

# 

start_sentence = 'There is no place like home.'
end_sentence_one_word = 'There is no place like Italy.'
end_sentence_keep_meaning = 'No place is like home.'
end_sentence_different = 'The cat jumps over the fence.'

encoded_sentence = encode_text_as_input(start_sentence)
encoded_sentence_1hot = nn.functional.one_hot(encoded_sentence, num_classes=bert_tokenizer.vocab_size).float()
start_input = encoded_sentence_1hot

embedding.eval()
start_output = embedding(encoded_sentence)
expected_output = start_output



# ---

perturbation_size = 0.1

input_parameter = start_input * (1. + perturbation_size * torch.rand(start_input.shape))

input_optimizer = InputOptimizer(bert_embedding_model, input_parameter).to(torch_device)

loss_function = nn.MSELoss(reduction='sum')
parameters_to_optimize = [ input_optimizer.input_vectors ]
optimizer = torch.optim.Adam(parameters_to_optimize, lr=0.0001)

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

for i in range(140):
  optimizer.zero_grad()
  current_output = input_optimizer()
  loss = loss_function(current_output, expected_output)
  loss.backward(retain_graph=True)
  #current_output.last_hidden_state.detach_()
  loss_values.append(loss.item())
  optimizer.step()


pyplot.plot(loss_values)
pyplot.show()


"""
1. extract the embedding matrix: only parameters of word_embedding (shape: nb_token X 768)
2. create a model from embedding where the word_embeddings layer is replaced by a linear map with the matrix
3. tokenize inputs then 1-hotize them



"""

"""
import numpy

x=torch.tensor(numpy.array([254, 18], dtype=int), dtype=torch.int32)
embedding.word_embeddings(x).shape
>   torch.Size([2, 768])

"""

























