import numpy as np
import pandas as pd


class RecommendationEngine(object):
    def __init__(self):
        pass

    def predict_score(self, user, item_id):
        raise NotImplementedError("Please implement the predict_score method in a subclass.")

    def make_recommendations(self, user, **kwargs):
        raise NotImplementedError("Please implement the make_recommendation method in a subclass.")


class RandomRecommendationEngine(RecommendationEngine):
    '''Random Baseline.'''

    def __init__(self):
        df_item = pd.read_json('data/generated/item.json')

    def predict_score(self, user, item_id):
        '''Returns a random score between 1.0 and 5.0.'''
        return np.random.random_sample() * 4 + 1

    def make_recommendations(self, user, **kwargs):
        '''Returns a randomly sampled item.'''
        return df_review.sample().iloc[0]['item_id']
