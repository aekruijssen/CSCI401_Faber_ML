# Importing the libraries
import matplotlib.pyplot as plt
import pickle
import os
import numpy as np
import pandas as pd
import faiss
from tqdm import tqdm

# LOAD DATA:

# read datasets
df_item = pd.read_json('saved/item.json')
df_user = pd.read_json('saved/user.json')

df_review = pd.read_csv('data/review.csv')


# PREPARATION:

# initialize the user index
d = len(df_user.iloc[0]['aspect_weights'])
index = faiss.IndexFlatIP(d)

# populate the index with user vectors
user_vectors = np.array(df_user['aspect_weights'].to_list()).astype(np.float32)
user_vectors = user_vectors / np.linalg.norm(user_vectors, axis=1, keepdims=True)
index.add(user_vectors)

def get_user_vector(reviews):
    # sample input: [{"item_id": "xyz", "rating": 5}]
    # output: user vector
    w_uk = np.zeros((d,))
    for review in reviews:
        item_row = df_item.loc[df_item["item_id"] == review["item_id"]]
        if len(item_row) == 0:
            continue
        w_uk += float(review["rating"]) * np.array(item_row.iloc[0]["aspect_weights"]).astype(float)
    return w_uk / np.linalg.norm(w_uk)

user_vector = get_user_vector([{"item_id": "XOSRcvtaKc_Q5H1SAzN20A", "rating": 5}])


# ITEM RATING PREDICTION:

def rate_item(user_vector, item_id, k=10):
    # fetch item vector
    item_vector = np.array(df_item[df_item["item_id"] == item_id].iloc[0]["aspect_weights"]).astype(float)

    # get the nearest k users given the user vector
    nearest_user_dists, nearest_users = index.search(user_vector.reshape((1,d)).astype(np.float32), k)
    nearest_user_dists = nearest_user_dists / np.linalg.norm(nearest_user_dists)

    score_info = []

    # for each nearby user...
    for user_ix in nearest_users[0]:
        # get his/her similarity value the queried user
        user_id = df_user.iloc[user_ix]["user_id"]
        nearby_vector = np.array(df_user[df_user["user_id"] == user_id].iloc[0]["aspect_weights"]).astype(float)
        user_sim = np.dot(user_vector, nearby_vector)/(np.linalg.norm(user_vector) * np.linalg.norm(nearby_vector))

        # iterate all his/her reviews
        reviews = df_review[df_review["user_id"] == user_id]
        for _, review in reviews.iterrows():
            # record the (user similarity, item similarity, reviewed score) tuple
            item = df_item[df_item["item_id"] == review["business_id"]]
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

rate_item(user_vector, "XOSRcvtaKc_Q5H1SAzN20A")

def predict(user, item_id):
    return rate_item(get_user_vector(user["reviews"]), item_id)
