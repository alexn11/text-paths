import torch
from torch import nn

from transformers import BertModel
from transformers import BertTokenizer

from input_optimizer import ModelInverter


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
    y = self.layer_normalization(y)
    return y



class BertModules:

  def __init__(self, bert_model_name = 'bert-base-uncased'):
    self.model_name = bert_model_name
    
    self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
    self.input_length = self.tokenizer.max_model_input_sizes[self.model_name]
    self.max_token = len(self.tokenizer.vocab)

    self.bert_model = BertModel.from_pretrained(self.model_name)

    self.core_model = self.bert_model.encoder
    
    self.embedding = self.bert_model.embeddings

    word_embedding = self.embedding.word_embeddings
    position_embedding = self.embedding.position_embeddings
    token_type_embedding = self.embedding.token_type_embeddings
    self.embedding_layer_norm = self.embedding.LayerNorm
    
    self.embedding_matrix = word_embedding.state_dict()['weight']
    self.position_embedding_vector = position_embedding.state_dict()['weight']
    self.token_type_vector = token_type_embedding.state_dict()['weight'][0,:]
    

    
  def get_core_model(self):
    return self.core_model
    
  def get_embeddings(self):
    return self.embedding
  
  def get_embedding_parameters(self):
    return self.embedding_matrix, self.position_embedding_vector, self.token_type_vector
    
  def get_embedding_model(self):
    self.embedding_model = BertEmbeddings1Hot(self.embedding_matrix,
               self.token_type_vector,
               self.position_embedding_vector,
               self.embedding_layer_norm)
    return self.embedding_model

  def encode_text_as_input(self, text):
    return self.tokenizer.encode(text,
                                 return_tensors='pt',
                                 max_length=self.input_length,
                                 pad_to_max_length=True)

  def encode_sequence_as_1hot(self, token_sequence):
    return nn.functional.one_hot(token_sequence,
                                 num_classes=self.tokenizer.vocab_size).float()
    
  def eval_embedding(self, text):
    encoded_input = self.encode_text_as_input(text)
    return self.embedding(encoded_input)

  def compute_core_model_output(self, text):
    core_input = self.eval_embedding(text)
    core_output = self.core_model(core_input).last_hidden_state
    return core_output
    
  def compute_text_steps(self, text):
    token_sequence = self.encode_text_as_input(text)
    sequence_1hot = self.encode_sequence_as_1hot(token_sequence)
    embedding_output = self.embedding(token_sequence)
    core_output = self.core_model(embedding_output).last_hidden_state
    return token_sequence, sequence_1hot, embedding_output, core_output
    
    



def core_bert_loss_function(output, target, loss_function=nn.MSELoss(reduction='sum')):
  #return loss_function(output.last_hidden_state, target.last_hidden_state)
  return loss_function(output.last_hidden_state, target)










































