import os
import numpy as np
import pandas as pd
from sklearn.utils import shuffle


def get_dataset_by_name(dataset_name):
    if dataset_name == 'yelp':
        return YelpReviewsDataset
    else:
        raise ValueError("Dataset with name {} doesn't exist.".format(dataset_name))


class Dataset(object):
    def __init__(self, train_prob=0.6, val_prob=0.2, seed=42):
        self.train_prob = 0.6
        self.val_prob = 0.2
        self.test_prob = 1.0 - self.train_prob - self.val_prob
        self.seed = seed
    
    def get_dataset_split(self, mode='train'):
        raise NotImplementedError('Please implement the get_dataset_split method in a subclass.')
        
    def sample(self, mode='train'):
        self.sample_batch(1, mode)[0]
        
    def sample_batch(self, batch_size, mode='train'):
        raise NotImplementedError('Please implement the sample_batch method in a subclass.')
        
    def size(self, mode='train'):
        raise NotImplementedError('Please implement the size method in a subclass.')
        
    def get_ids(self, mode='train'):
        raise NotImplementedError('Please implement the get_ids method in a subclass.')
        
    def __getitem__(self, id):
        raise NotImplementedError('Please implement indexing in a subclass.')


class YelpReviewsDataset(Dataset):
    def __init__(self, **kwargs):
        super(YelpReviewsDataset, self).__init__(**kwargs)

        # read dataset
        self.dataset = shuffle(pd.read_json('data/generated/review.json'), random_state=self.seed)
        
        self.dataset_train = self.dataset.iloc[:self.size('train')]
        self.dataset_val = self.dataset.iloc[self.size('train'):self.size('train') + self.size('val')]
        self.dataset_test = self.dataset.iloc[-self.size('test'):]
    
    def get_dataset(self, mode='train'):
        if mode == 'train':
            return self.dataset_train
        elif mode == 'val':
            return self.dataset_val
        elif mode == 'test':
            return self.dataset_test
    
    def sample_batch(self, batch_size, mode='train'):
        sampled_rows = self.get_dataset(mode).sample(batch_size)
        return self._df_to_data_point(sampled_rows)
    
    def size(self, mode='train'):
        if mode == 'train':
            return int(len(self.dataset) * self.train_prob)
        elif mode == 'val':
            return int(len(self.dataset) * self.val_prob)
        elif mode == 'test':
            return len(self.dataset) - self.size('train') - self.size('val')

        raise ValueError('Please enter a valid mode.')
        
    def get_ids(self, mode='train'):
        if mode == 'train':
            return list(range(self.size('train')))
        elif mode == 'val':
            return list(range(self.size('train'), self.size('train') + self.size('val')))
        elif mode == 'test':
            return list(range(self.size('train') + self.size('val'), len(self.dataset)))

        raise ValueError('Please enter a valid mode.')
        
    def __getitem__(self, key):
        return self._df_to_data_point(self.dataset.iloc[key:key+1])[0]
        
    def _df_to_data_point(self, df):
        data_points = []
        for index, row in df.iterrows():
            reviews_of_user = self.dataset_train.loc[self.dataset_train['user_id'] == row['user_id']]
            reviews_of_user = reviews_of_user[['business_id', 'stars']]
            user_obj = {"reviews": [], "user_id": row['user_id'], "vec": row['user_vec']}
            for _, review in reviews_of_user.iterrows():
                user_obj["reviews"].append({'business_id': review['business_id'], 'rating': review['stars']})
            data_points.append({
                "user": user_obj,
                "item_id": row['business_id'],
                "item_vec": row['business_vec'],
                "stars": row['stars']
            })
        return data_points
