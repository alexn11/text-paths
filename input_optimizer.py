
from calendar import c
from re import L
from regex import E
import torch
from torch import nn


class InputOptimizer(nn.Module):
  def __init__(self, model, input_vectors):
    super(InputOptimizer, self).__init__()
    self.model = model
    self.input_vectors = nn.parameter.Parameter(input_vectors)
    #self.register_parameter('input_vector', self.input_vectors)
  def forward(self):
    return self.model(self.input_vectors)
  def to(self, device):
    super(InputOptimizer, self).to(device)
    self.model.to(device)
    self.input_vectors = self.input_vectors.to(device)

class GradientDescent(torch.optim.Optimizer):
  def __init__(self, params, lr=1.):
    defaults = {
      'lr': lr,
    }
    super(GradientDescent, self).__init__(params, defaults)

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

  def __init__(self, model, input_vectors, loss_function=nn.MSELoss(reduction='sum'), tensorboard_writer=None):
    self.input_optimizer_model = InputOptimizer(model, input_vectors)
    self.loss_function = loss_function
    self.tensorboard_writer = tensorboard_writer

  def to(self, device):
    self.input_optimizer_model.to(device)
    
  def log_value(self, name, index=0, value=None):
    if(self.tensorboard_writer is None):
      return
    print(f'log_value {name}: {index} - {value}')
    self.tensorboard_writer.add_scalar(name, value, index)
    self.tensorboard_writer.flush()
    
  def prepare_computations(self, expected_output, torch_device='cpu', lr=0.0001, optimizer=None):
  
    self.expected_output = expected_output.to(torch_device)
  
    self.input_optimizer_model.to(torch_device)
    parameters_to_optimize = [ self.input_optimizer_model.input_vectors ]
    
    if(optimizer is None):
      self.optimizer = torch.optim.Adam(parameters_to_optimize, lr=lr)
      #self.optimizer = torch.optim.SGD(parameters_to_optimize, lr=lr)
      #self.optimizer = GradientDescent(parameters_to_optimize, lr=lr)
    else:
      self.optimizer = optimizer
    
    input_optimizer = self.input_optimizer_model
    input_optimizer.train()
    
  def computation_step(self, step=None):
    self.optimizer.zero_grad()
    current_output = self.input_optimizer_model()
    loss = self.loss_function(current_output, self.expected_output)
    #print(current_output)
    #print(self.expected_output)
    #print(self.input_optimizer_model.input_vectors)
    #print(list(self.input_optimizer_model.model.parameters())[0])
    #print(loss)
    #print(loss.grad_fn)
    loss.backward(retain_graph=True)
    loss_value = loss.item()
    self.log_value('loss', index=step, value=loss_value)
    self.optimizer.step()
    return loss_value
    
  def compute_inverse(self,
                      expected_output,
                      n_max_steps=1,
                      min_loss=0.1,
                      torch_device='cpu',
                      lr=0.0001,
                      optimizer=None):

    self.prepare_computations(expected_output, torch_device=torch_device, lr=lr, optimizer=optimizer)
  
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






































































