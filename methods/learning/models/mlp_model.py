import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from util.pytorch import to_tensor
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

    def process_input(self, batch):
        config = self._config
        user_vec = np.array([item['user']['vec'] for item in batch])
        item_vec = np.array([item['item']['vec'] for item in batch])
        labels = np.array([item['stars'] for item in batch])
        user_vec = to_tensor(user_vec, config.device)
        item_vec = to_tensor(item_vec, config.device)
        labels = to_tensor(labels, config.device)
        return (user_vec, item_vec), labels
