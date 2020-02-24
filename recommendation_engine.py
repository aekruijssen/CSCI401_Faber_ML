import os
import numpy as np
import pandas as pd
import faiss

class RecommendationEngine(object):
    def __init__(self):
        # print message
        print('Initializing recommendation engine...')
        
        # read datasets
        self.df_item = pd.read_json('saved/item.json')
        self.df_user = pd.read_json('saved/user.json')
        self.df_review = pd.read_csv('data/review.csv')
        
        # initialize the user index
        self.d = len(self.df_user.iloc[0]['aspect_weights'])
        self.index = faiss.IndexFlatIP(self.d)

        # populate the index with user vectors
        user_vectors = np.array(self.df_user['aspect_weights'].to_list()).astype(np.float32)
        user_vectors = user_vectors / np.linalg.norm(user_vectors, axis=1, keepdims=True)
        self.index.add(user_vectors)
        
        # print message
        print('Recommendation engine initialization finished.')
        
    def get_user_vector(self, reviews):
        # sample input: [{"item_id": "xyz", "rating": 5}]
        # output: user vector
        w_uk = np.zeros((self.d,))
        for review in reviews:
            item_row = self.df_item.loc[self.df_item["item_id"] == review["item_id"]]
            if len(item_row) == 0:
                continue
            w_uk += float(review["rating"]) * np.array(item_row.iloc[0]["aspect_weights"]).astype(float)
        return w_uk / np.linalg.norm(w_uk)
    
    def rate_item(self, user_vector, item_id, k=10):
        # fetch item vector
        item_vector = np.array(self.df_item[self.df_item["item_id"] == item_id].iloc[0]["aspect_weights"]).astype(float)

        # get the nearest k users given the user vector
        nearest_user_dists, nearest_users = self.index.search(user_vector.reshape((1,self.d)).astype(np.float32), k)
        nearest_user_dists = nearest_user_dists / np.linalg.norm(nearest_user_dists)

        score_info = []

        # for each nearby user...
        for user_ix in nearest_users[0]:
            # get his/her similarity value the queried user
            user_id = self.df_user.iloc[user_ix]["user_id"]
            nearby_vector = np.array(self.df_user[self.df_user["user_id"] == user_id].iloc[0]["aspect_weights"]).astype(float)
            user_sim = np.dot(user_vector, nearby_vector)/(np.linalg.norm(user_vector) * np.linalg.norm(nearby_vector))

            # iterate all his/her reviews
            reviews = self.df_review[self.df_review["user_id"] == user_id]
            for _, review in reviews.iterrows():
                # record the (user similarity, item similarity, reviewed score) tuple
                item = self.df_item[self.df_item["item_id"] == review["business_id"]]
                if len(item) == 0:
                    continue
                review_vector = np.array(item.iloc[0]["aspect_weights"]).astype(float)
                item_sim = np.dot(item_vector, review_vector)/(np.linalg.norm(item_vector) * np.linalg.norm(review_vector))
                score_info.append([user_sim, item_sim, review["stars"]])

        # accumulate results
        weights = np.array([x[0] * x[1] for x in score_info])
        weights /= np.sum(weights)
        ratings = np.array([x[2] for x in score_info])
        return weights.dot(ratings)
    
    def predict_score(self, user, item_id):
        # sample user input: { 
        #     "latitude": 40,
        #     "longitude": -80,
        #     "...": (other info),
        #     "reviews": [
        #         {
        #             "item_id": "A",
        #             "rating": 4.5
        #         }
        #     ]
        # }
        return self.rate_item(self.get_user_vector(user["reviews"]), item_id)