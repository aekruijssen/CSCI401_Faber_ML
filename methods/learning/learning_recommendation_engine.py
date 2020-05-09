import os
import torch
import numpy as np
import pandas as pd
import wandb

from methods.learning.util import AttrDict
from methods.learning.models import get_model_by_name
from methods.learning.util.pytorch import get_ckpt_path
from components.recommendation_engine import RecommendationEngine


class LearningRecommendationEngine(RecommendationEngine):
    '''Deep learning based recommendation.'''
    def __init__(self, run_name='yelp.lstm_v1'):
        print('Loading recommendation engine...')

        self.wandb_api = wandb.Api()
        run = self.wandb_api.run("jingyuny/faber/{}".format(run_name))
        self.config = AttrDict(run.config)

        self.device = self._setup_device(self.config.gpu)
        self.config.device = self.device
        self.df = pd.read_json('data/generated/review.json')
        self.model = get_model_by_name(self.config.model)(self.config)
        self.model = self.model.to(self.device)

        # loading checkpoint
        log_dir = os.path.join('./logs/', run_name)
        ckpt_path, step = get_ckpt_path(log_dir, step=None)
        print('Loading checkpoint from {}...'.format(ckpt_path))
        ckpt = torch.load(ckpt_path)
        self.model.load_state_dict(ckpt['model'])
        print('Loaded checkpoint at step {}.'.format(ckpt['step']))

        print('Loading finished.')

    def predict_score(self, user, item_id):
        # sample user input: {
        #     "location": {
        #         "latitude": 40,
        #         "longitude": -80,
        #     },
        #     "...": (other info),
        #     "reviews": [
        #         {
        #             "business_id": "A",
        #             "text": "blah",
        #             "rating": 4.5
        #         }
        #     ]
        # }
        reviews_of_item = self.df.loc[self.df['business_id'] == item_id]

        data_point = {
            "user": {
                "reviews": [x["content"] for x in user['reviews'] if 'content' in x.keys()]
            },
            "item": {
                "reviews": reviews_of_item['text'].tolist()
            },
            "stars": 0
        }

        inputs, _ = self.model.process_input([data_point])

        pred_labels = self.model(*inputs)

        pred_label = pred_labels.detach().cpu().numpy().flatten()[0]

        return float(pred_label)

    def predict_score_v2(self, user, item):
        data_point = {
            "user": {
                "reviews": [x["content"] for x in user['reviews'] if 'content' in x.keys()]
            },
            "item": {
                "reviews": [x["content"] for x in item['reviews'] if 'content' in x.keys()]
            },
            "stars": 0
        }

        inputs, _ = self.model.process_input([data_point])

        pred_labels = self.model(*inputs)

        return pred_labels.detach().cpu().numpy().flatten()

    def make_recommendations(self, user):
        raise NotImplementedError()

    def _setup_device(self, gpu):
        if gpu is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu)
            assert torch.cuda.is_available()
            return torch.device("cuda")
        else:
            return torch.device("cpu")
