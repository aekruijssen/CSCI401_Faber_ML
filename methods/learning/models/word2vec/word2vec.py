import numpy as np
import string
from tqdm import tqdm


class Word2Vec(object):
    def __init__(self, config):
        self.embedding_dim = config.embedding_dim
        self.embedding_dict = {}
        self.read_dict()

    def read_dict(self):
        print('Reading word2vec embeddings...')
        with open("methods/learning/models/word2vec/glove.6B.{}d.txt".format(self.embedding_dim), 'r', encoding="utf-8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], np.float32)
                self.embedding_dict[word] = vector
        print('Read {} words.'.format(len(self.embedding_dict)))

    def process(self, sentence):
        sentence = sentence.translate(str.maketrans(dict.fromkeys(string.punctuation)))
        words = sentence.lower().split()
        embeddings = []
        for word in words:
            if word in self.embedding_dict.keys():
                embeddings.append(self.embedding_dict[word])
        if len(embeddings) == 0:
            return None
        return np.stack(embeddings)
