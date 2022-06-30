import torch

def remove_last_layer(inception_model):
    # https://stackoverflow.com/questions/52548174/how-to-remove-the-last-fc-layer-from-a-resnet-model-in-pytorch
    inception_model.fc = torch.nn.Identity()
    return inception_model


class PixelDifferencesLoss(torch.nn.Module):

    def __init__(self, input_loss_weight=1., tensorboard_writer=None):
        super(PixelDifferencesLoss, self).__init__()
        self.input_loss_weight = input_loss_weight
        self.tensorboard_writer = tensorboard_writer
        if(self.tensorboard_writer is not None):
            self.step = 0
        self.compute_mse_loss = torch.nn.MSELoss(reduce='sum')

    def compute_pixel_differences_L2_norm(self, x):
        # x: batch_size x 3 x 299 x 299
        # the number of diagonal differences is slightly less than horiz/vert differences
        diagonal_factor = 1. / (2.**(1/2))
        squared_differences = (x[:, :, 1:, :] - x[:, :, :-1, :]).mean((1, 2, 3))
        squared_differences += (x[:, :, :, 1:] - x[:, :, :, :-1]).mean((1, 2, 3))
        squared_differences += diagonal_factor * ((x[:, :, 1:, 1:] - x[:, :, :-1, :-1]).mean((1, 2, 3)))
        squared_differences += diagonal_factor * ((x[:, :, 1:, :-1] - x[:, :, :-1, 1:]).mean((1, 2, 3)))
        return squared_differences.pow(2).sum()

    def compute_input_loss(self, inputs):
        #self.compute_pixel_differences_L2_norm(inputs)
        return torch.linalg.vector_norm(inputs, ord=1)

    def forward(self, model_outputs, expected_outputs, inputs):
        mse_loss = self.compute_mse_loss(model_outputs, expected_outputs)
        differences_loss = self.input_loss_weight * self.compute_input_loss(inputs)
        if(self.tensorboard_writer is not None):
            self.tensorboard_writer.add_scalar('PDL/MSE', mse_loss.item(), self.step)
            self.tensorboard_writer.add_scalar('PDL/inputs', differences_loss.item(), self.step)
            self.tensorboard_writer.flush()
            self.step += 1
        return mse_loss + differences_loss





