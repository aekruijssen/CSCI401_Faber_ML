import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils import MLP


class MlpModel(nn.Module):
    def __init__(self, config):
        ''' MLP model that maps user and item vectors to a score.
            Config should include info about n_hid, h_dim, input_dim, output_min, output_max
        '''
        super().__init__()
        self._config = config 
        self.fc = MLP(2 * config.input_dim, 1, [config.h_dim] * config.n_hid)

    def forward(self, user, item):
        output_min, output_max = self._config.output_min, self._config.output_max
        x = torch.cat((user, item), 1)
        x = self.fc(x)
        x = F.tanh(x) * (output_max - output_min) + output_min
        return x
