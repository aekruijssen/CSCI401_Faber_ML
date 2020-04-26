import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def fanin_init(tensor):
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = np.prod(size[1:])
    else:
        raise Exception("Shape must be have dimension at least 2.")
    bound = 1. / np.sqrt(fan_in)
    return tensor.data.uniform_(-bound, bound)


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hid_dims=[]):
        super().__init__()
        activation_fn = nn.ReLU()

        fc = []
        prev_dim = input_dim
        for d in hid_dims:
            fc.append(nn.Linear(prev_dim, d))
            fanin_init(fc[-1].weight)
            fc[-1].bias.data.fill_(0.1)
            fc.append(activation_fn)
            prev_dim = d
        fc.append(nn.Linear(prev_dim, output_dim))
        fc[-1].weight.data.uniform_(-1e-3, 1e-3)
        fc[-1].bias.data.uniform_(-1e-3, 1e-3)
        self.fc = nn.Sequential(*fc)

    def forward(self, ob):
        return self.fc(ob)


class SentenceEncoderLSTM(nn.Module):
    '''
    Embeds a sentence to a sentence vector. A sentence is represented by a
    vector of size (sentence_length, embedding_size).
    '''
    def __init__(self, config):
        super().__init__()

        self.embedding_dim = config.embedding_dim
        self.hidden_dim = config.hidden_dim

        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, batch_first=True)

    def forward(self, sentence, lens):
        lstm_out, _ = self.lstm(sentence)
        return lstm_out[torch.arange(lens.shape[0]), lens - 1]

