import sys
import argparse

import numpy

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

from input_optimizer import ModelInverter
from inception_tools import remove_last_layer, PixelDifferencesLoss

# weight decay 0 -> max steps 20000+
# wieght decay 0.01 -> max steps: 1000
# weight decay > 0.1 -> max steps ~500


# smoothin-decay 
# 1.e6 - 0.001 < class < differences
# < embedd

argparser = argparse.ArgumentParser()
argparser.add_argument('--model-type', type=str, default='classes')
argparser.add_argument('--image-name', type=str, default='')
argparser.add_argument('--image-class', type=int, default=18)
argparser.add_argument('--device', type=str, default='cuda')
argparser.add_argument('--weight-decay', type=float, default=0.5) # 0.01
argparser.add_argument('--inverter-nb-max-steps', type=int, default=500) # 1000 # 20000
argparser.add_argument('--do-save-initial-state', type=bool, default=False)
argparser.add_argument('--loss-type', type=str, default='SmoothedMSE')
argparser.add_argument('--smoothing', type=float, default=1.e-7)
argparser.add_argument('--embedding-init', type=str, default='index')
argparser.add_argument('--embedding-index', type=int, default=0)
parsed_arguments = argparser.parse_args()

assert(parsed_arguments.device in [ 'cuda', 'cpu' ])
assert(parsed_arguments.smoothing >= 0.0)
assert(parsed_arguments.embedding_init in [ 'random', 'index' ])
assert(0 <= parsed_arguments.embedding_index < 2048)
assert(parsed_arguments.model_type in [ 'embedding', 'classes' ])

print('command line arguments values:')
print(parsed_arguments)

assert(parsed_arguments.loss_type in [ 'MSE', 'SmoothedMSE' ])

image_name = parsed_arguments.image_name
model_type = parsed_arguments.model_type

if(model_type == 'classes'):
    image_class = parsed_arguments.image_class
    assert(image_class >= 0)
    assert(image_class < 1000)
    if(image_name == ''):
        image_name = f'{image_class}'
elif(model_type == 'embedding'):
    embedding_init = parsed_arguments.embedding_init
    if(embedding_init == 'index'):
        embedding_index = parsed_arguments.embedding_index

if(image_name == ''):
    image_name = 'noname'

weight_decay = parsed_arguments.weight_decay
inverter_nb_max_steps = parsed_arguments.inverter_nb_max_steps
do_save_initial_state = parsed_arguments.do_save_initial_state
loss_type = parsed_arguments.loss_type
smoothing = parsed_arguments.smoothing

if(parsed_arguments.device == 'cuda'):
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {torch_device} device')

tensorboard_writer = SummaryWriter()

model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)

if(model_type == 'embedding'):
    print(f'model type is {model_type}: removing the last layers')
    model = remove_last_layer(model)
model.to(torch_device)

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

if(do_save_initial_state):
    save_image(input_vector[0], f'initial-image-{image_name}.png')

model.eval()

initial_outputs = model(input_vector)
# input shape: batch_size x 3 x 299 x 299
# output shape: batch_size x 1000

expected_output_vector = torch.zeros_like(initial_outputs)

if(model_type == 'classes'):
    expected_output_vector[0, image_class] = 1.
else:
    if(embedding_init == 'random'):
        expected_output_vector += torch.rand_like(expected_output_vector)
    else:
        expected_output_vector[0, embedding_index] = 1.

if(loss_type == 'MSE'):
    loss_function = nn.MSELoss(reduction='sum')
    do_use_inputs_in_loss = False
elif(loss_type == 'SmoothedMSE'):
    loss_function = PixelDifferencesLoss(
        input_loss_weight=smoothing,
        tensorboard_writer=tensorboard_writer)
    do_use_inputs_in_loss = True

model_inverter = ModelInverter(model,
                               input_vector,
                               loss_function=loss_function,
                               do_use_inputs_in_loss_function=do_use_inputs_in_loss,
                               torch_device=torch_device,
                               tensorboard_writer=tensorboard_writer)

loss_history = model_inverter.compute_inverse(expected_output_vector,
                                              n_max_steps=inverter_nb_max_steps,
                                              lr=0.1,
                                              optimizer_kwargs={
                                                'weight_decay': weight_decay,
                                              })
# lr experiments:
# 0.001 smooth slow
# 0.1 faster still smooth slow < best
# 1 slightly worse than 0.1
# 0.2 indistin


print('generating image')
generated_image = model_inverter.get_computed_solution()

#output = model(generated_image)
#print(output)
#print(expected_output_vector)

generated_image_file_name = f'generated-image-{image_name}.png'
print(f'saving image to "{generated_image_file_name}"')
save_image(generated_image[0], generated_image_file_name)
print('done')

