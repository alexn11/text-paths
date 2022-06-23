
from calendar import c
from re import L
from regex import E
import torch
from torch import nn

# for debugging
import transformers
# >>>

class InputOptimizer(nn.Module):
  def __init__(self, model, input_vectors):
    super(InputOptimizer, self).__init__()
    self.model = model
    # necessary?🐈️
    for param in self.model.parameters(): # 🐈️
      param.requires_grad = False # 🐈️
    # 
    self.input_vectors = nn.parameter.Parameter(input_vectors)
    #self.register_parameter('input_vector', self.input_vectors)
    self.device = 'cpu'

  def forward(self):
    return self.model(self.input_vectors)
    #return self.model(self.input_vectors).last_hidden_state
    # DEBUG:
    #y = self.model(self.input_vectors)
    #y = y.last_hidden_state
    #y = y.to(self.device)
    #return y

  def to(self, device):
    super(InputOptimizer, self).to(device)
    self.model.to(device)
    self.input_vectors = self.input_vectors.to(device)
    self.device = device

  def train(self, mode=True):
    super(InputOptimizer, self).train(mode=mode)
    self.model.eval()

class GradientDescent(torch.optim.Optimizer):
  # TODO: regularization
  def __init__(self, params, lr=1., regularization_coeff=0.):
    defaults = {
      'lr': lr,
      'regularization_coeff': regularization_coeff,
    }
    super(GradientDescent, self).__init__(params, defaults)
    self.do_regularize = (regularization_coeff != 0.)

  def step(self, closure=None):

    loss = None
    if(closure is not None):
      with torch.enable_grad():
        loss = closure()

    for group in self.param_groups:
      lr = group['lr']
      for p in group['params']:
        if(p.grad is not None):
          p.data.add_(p.grad.data, alpha=lr)
    return loss


class ModelInverter:

  def __init__(self,
               model,
               input_vectors,
               loss_function=nn.MSELoss(reduction='sum'),
               torch_device='cpu',
               tensorboard_writer=None):
    self.torch_device = torch_device
    self.input_optimizer_model = InputOptimizer(model, input_vectors)
    self.loss_function = loss_function
    self.tensorboard_writer = tensorboard_writer
    self.log_folder = ''

  #def to(self, device):
  #  self.input_optimizer_model.to(device)
    
  def log_value(self, folder, name, index=0, value=None):
    if(self.tensorboard_writer is None):
      return
    #print(f'log_value {folder}/{name}: {index} - {value}')
    if(folder != ''):
      name = f'{folder}/{name}'
    self.tensorboard_writer.add_scalar(name, value, index)
    self.tensorboard_writer.flush()
    
  def move_data_and_model_to_device(self):
    self.expected_output = self.expected_output.to(self.torch_device)
    self.input_optimizer_model.to(self.torch_device)

  def prepare_computations(self,
                           expected_output,
                           lr=0.0001,
                           optimizer_class=None,
                           optimizer_kwargs={}):
  
    self.expected_output = expected_output

    self.move_data_and_model_to_device()
  
    parameters_to_optimize = [ self.input_optimizer_model.input_vectors ]
    #assert(parameters_to_optimize[0].device == torch.device('cuda:0'))
    
    if(optimizer_class is None):
      # add some regularisation
      # prbably not the best way to pass arguments...
      weight_decay = optimizer_kwargs.get('weight_decay', 1.e-5)
      self.optimizer = torch.optim.Adam(parameters_to_optimize, lr=lr, weight_decay=weight_decay)
      #self.optimizer = torch.optim.SGD(parameters_to_optimize, lr=lr)
      #self.optimizer = GradientDescent(parameters_to_optimize, lr=lr)
    else:
      self.optimizer = optimizer_class(parameters_to_optimize, lr=lr, **optimizer_kwargs)
    
    input_optimizer = self.input_optimizer_model
    input_optimizer.train()
    
  def computation_step(self, step=None):
    self.optimizer.zero_grad()
    current_output = self.input_optimizer_model.forward()
    loss = self.loss_function(current_output, self.expected_output)
    #print(self.input_optimizer_model.model)
    #print('devices:')
    #print(f'  cur_outp:{current_output.device},')
    #print(f'  exp outp: {self.expected_output.device}')
    #print(f'  opt inp v: {self.input_optimizer_model.input_vectors.device}')
    #print(f'  loss: {loss.device}')
    #assert(current_output.device == torch.device('cuda:0'))
    #assert(self.expected_output.device == torch.device('cuda:0'))
    #assert(self.input_optimizer_model.input_vectors.device == torch.device('cuda:0'))
    #for i, p in enumerate(list(self.input_optimizer_model.model.parameters())):
    #  assert(p.device == torch.device('cuda:0'))
    #assert(loss.device == torch.device('cuda:0'))
    #print(loss.grad_fn)
    #loss.backward(retain_graph=True) # what does retain_graph do?
    #self.getBack(loss.grad_fn)
    loss.backward(retain_graph=True)
    self.optimizer.step()
    loss_value = loss.item()
    self.log_value(self.log_folder, 'loss', index=step, value=loss_value)
    return loss_value
    
  def compute_inverse(self,
                      expected_output,
                      n_max_steps=1,
                      min_loss=0.1,
                      torch_device='cpu',
                      lr=0.0001,
                      optimizer_class=None,
                      optimizer_kwargs={},
                      log_folder=''):

    self.log_folder = log_folder

    self.prepare_computations(expected_output,
                              lr=lr,
                              optimizer_class=optimizer_class,
                              optimizer_kwargs=optimizer_kwargs)
  
    loss_values = []
    
    step = 0
    while(step < n_max_steps):
      loss_value = self.computation_step(step=step)
      loss_values.append(loss_value)
      step += 1
      if(loss_value < min_loss):
        break

    return loss_values

  def get_computed_solution(self):
    return self.input_optimizer_model.input_vectors



