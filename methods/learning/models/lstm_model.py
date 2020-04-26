import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import numpy as np

from util import AttrDict
from util.pytorch import to_tensor
from models.word2vec.word2vec import Word2Vec
from models.utils import SentenceEncoderLSTM
from models.mlp_model import MlpModel


class LSTMModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._config = config

        self.word2vec = Word2Vec(AttrDict({ "embedding_dim": config.embedding_dim }))
        self.sentence_lstm = SentenceEncoderLSTM(AttrDict({
            "embedding_dim": config.embedding_dim,
            "hidden_dim": config.h_dim
        }))
        self.mlp = MlpModel(AttrDict({
            "input_dim": config.h_dim,
            "h_dim": config.h_dim,
            "n_hid": config.n_hid,
            "output_min": config.output_min,
            "output_max": config.output_max
        }))

    def forward(self, user, item):
        return self.mlp(user, item)
    
    def process_input(self, batch):
        config = self._config
        user_vecs, item_vecs = [], []
        for i in range(len(batch)):
            user_reviews = batch[i]['user']['reviews']
            item_reviews = batch[i]['item']['reviews']
            n_ur, n_ir = len(user_reviews), len(item_reviews)

            reviews = user_reviews + item_reviews
            user_review_embeddings = []
            user_review_lens = []
            item_review_embeddings = []
            item_review_lens = []
            for j, review in enumerate(reviews):
                embedding = self.word2vec.process(review)
                if embedding is None:
                    continue
                embedding = to_tensor(embedding, config.device)
                if j < n_ur:
                    user_review_embeddings.append(embedding)
                    user_review_lens.append(embedding.shape[0])
                else:
                    item_review_embeddings.append(embedding)
                    item_review_lens.append(embedding.shape[0])

            if len(user_review_embeddings) > 0:
                user_review_lens = to_tensor(user_review_lens, config.device).type(torch.int64)
                user_review_embeddings = pad_sequence(user_review_embeddings, batch_first=True)
                user_vec = self.sentence_lstm(user_review_embeddings, user_review_lens)
            else:
                user_vec = to_tensor(np.zeros((config.h_dim,)), config.device)

            if len(item_review_embeddings) > 0:
                item_review_lens = to_tensor(item_review_lens, config.device).type(torch.int64)
                item_review_embeddings = pad_sequence(item_review_embeddings, batch_first=True)
                item_vec = self.sentence_lstm(item_review_embeddings, item_review_lens)
            else:
                item_vec = to_tensor(np.zeros((config.h_dim,)), config.device)

            user_vecs.append(user_vec)
            item_vecs.append(item_vec)

        labels = np.array([item['stars'] for item in batch])
        labels = to_tensor(labels, config.device)

        return (torch.stack(user_vecs), torch.stack(item_vecs)), labels
