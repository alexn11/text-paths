
import numpy
import torch
from torch import nn
from input_optimizer import ModelInverter
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

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
                                              n_max_steps=2000,
                                              torch_device=torch_device,
                                              lr=0.001)




generated_image = model_inverter.get_computed_solution()
output = model(generated_image)
print(output)
print(expected_output_vector)

save_image(generated_image[0], 'generated-image.png')