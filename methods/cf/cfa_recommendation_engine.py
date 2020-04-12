import numpy as np
import pandas as pd
import faiss
from shapely.geometry import Point
import requests
from pathos.multiprocessing import ProcessPool as Pool

from components.recomemndation_engine import RecommendationEngine


BASE_URL = 'http://lim-c.usc.edu:6010/'

def request_item_rating(user, item):
    json_data = { "user": user, "business_id": item }
    req = requests.post(BASE_URL + 'predict_api', json=json_data)
    print('finish with: {}'.format(req.text))
    return item, float(req.text)

class CfaRecommendationEngine(RecommendationEngine):
    '''Collaborative Filtering from Aspect Vectors.'''

    def __init__(self, df_review=None, relative=True):
        # print message
        print('Initializing recommendation engine...')

        # read datasets
        self.df_item = pd.read_json('data/generated/item.json')
        self.df_user = pd.read_json('data/generated/user.json')
        if df_review is not None:
            self.df_review = df_review
        else:
            self.df_review = pd.read_csv('data/yelp_data/review.csv')

        # flag to specify either to normalize ratings on user review average
        self.relative = relative

        # initialize the user index
        self.d = len(self.df_user.iloc[0]['aspect_weights'])
        self.user_index = faiss.IndexFlatIP(self.d)

        # populate the user index with user vectors
        user_vectors = np.array(self.df_user['aspect_weights'].to_list()).astype(np.float32)
        user_vectors = user_vectors / np.linalg.norm(user_vectors, axis=1, keepdims=True)
        self.user_index.add(user_vectors)

        # build geo-dataframe for items
        geometry = [Point(xy) for xy in zip(self.df_item['longitude'], self.df_item['latitude'])]
        self.geo_df_item = gpd.GeoDataFrame(self.df_item, geometry=geometry)

        print('Recommendation engine initialization finished.')

    def get_user_vector(self, reviews):
        # sample input: [{"business_id": "xyz", "rating": 5}]
        # output: user vector
        w_uk = np.zeros((self.d,))
        for review in reviews:
            item_row = self.df_item.loc[self.df_item["item_id"] == review["business_id"]]
            if len(item_row) == 0:
                continue
            w_uk += float(review["rating"]) * np.array(item_row.iloc[0]["aspect_weights"]).astype(float)
        return w_uk / np.linalg.norm(w_uk)

    def get_user_rating_avg(self, user_id):
        filtered_reviews = self.df_review[self.df_review['user_id'] == user_id]
        return np.mean(filtered_reviews['stars'].tolist())

    def rate_item(self, user_vector, user_rating_avg, item_id, k=10):
        # fetch item vector
        item_vector = np.array(self.df_item[self.df_item["item_id"] == item_id].iloc[0]["aspect_weights"]).astype(float)

        # get the nearest k users given the user vector
        nearest_user_dists, nearest_users = self.user_index.search(user_vector.reshape((1,self.d)).astype(np.float32), k)
        nearest_user_dists = nearest_user_dists / np.linalg.norm(nearest_user_dists)

        score_info = []

        # for each nearby user...
        for user_ix in nearest_users[0]:
            # get his/her similarity value the queried user
            user_id = self.df_user.iloc[user_ix]["user_id"]
            nearby_vector = np.array(self.df_user[self.df_user["user_id"] == user_id].iloc[0]["aspect_weights"]).astype(float)
            user_sim = np.dot(user_vector, nearby_vector)/(np.linalg.norm(user_vector) * np.linalg.norm(nearby_vector))

            # get average rating for the user
            if self.relative:
                user_score_avg = self.get_user_rating_avg(user_id)

            # iterate all his/her reviews
            reviews = self.df_review[self.df_review["user_id"] == user_id]
            for _, review in reviews.iterrows():
                # record the (user similarity, item similarity, reviewed score) tuple
                item = self.df_item[self.df_item["item_id"] == review["business_id"]]
                if len(item) == 0:
                    continue
                review_vector = np.array(item.iloc[0]["aspect_weights"]).astype(float)
                item_sim = np.dot(item_vector, review_vector)/(np.linalg.norm(item_vector) * np.linalg.norm(review_vector))
                score = review["stars"]
                if self.relative:
                    score -= user_score_avg
                score_info.append([user_sim, item_sim, score])

        # accumulate results
        weights = np.array([x[0] * x[1] for x in score_info])
        weights /= np.sum(weights)
        ratings = np.array([x[2] for x in score_info])
        return weights.dot(ratings) + (user_rating_avg if self.relative else 0)

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
        #             "rating": 4.5
        #         }
        #     ]
        # }
        reviews = user["reviews"]
        if "ratings" in user.keys():
            reviews += user["ratings"]
        user_rating_avg = np.mean([item["rating"] for item in reviews])
        return self.rate_item(self.get_user_vector(reviews), user_rating_avg, item_id)

    def make_recommendations(self, user, k=20, distance_threshold=0.3):
        user_location = Point(user['location']['longitude'], user['location']['latitude'])
        boundary = user_location.buffer(distance_threshold)
        nearby_items = self.geo_df_item[self.geo_df_item.geometry.within(boundary)]
        nearby_items = nearby_items.sort_values(by='stars', ascending=False).iloc[:2*k]

        p = Pool(24)
        item_id_list = nearby_items['item_id'].tolist()
        score_list = p.map(request_item_rating, [user] * len(item_id_list), item_id_list)
        p.close()

        score_list.sort(key=lambda x: x[1])
        return [x[0] for x in score_list[-k:]]
