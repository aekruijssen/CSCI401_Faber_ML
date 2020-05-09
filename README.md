# CSCI401 Faber ML

## Setup

Create a virtual environment: `virtualenv -p $(which python3) venv`.

Activate the environment: `source venv/bin/activate`.

Install packages: `pip3 install -r requirements.txt`.

Download the Yelp dataset from [this link](https://www.kaggle.com/yelp-dataset/yelp-dataset/version/4). Save the CSV files `yelp_business.csv`, `yelp_review.csv` and `yelp_user.csv` to directory `data/yelp_data/` and rename to `business.csv`, `review.csv`, and `user.csv`.

Download the aspect annotation information [aspect\_restaurants.csv](http://ir.ii.uam.es/aspects/data/vocabularies/aspects_restaurants.zip), [lexicon\_restaurants.csv](http://ir.ii.uam.es/aspects/data/lexicons/lexicon_restaurants.zip), and [annotations\_voc\_yelp\_restaurants.txt](http://ir.ii.uam.es/aspects/data/annotations/voc/annotations_voc_yelp_restaurants.zip). Extract the zip files and save the enclosed files to `data/aspect/`.

Download the saved user vectors, item vectors, and processed reviews at [this link](https://drive.google.com/drive/folders/1Jt3U2ix-zsZljOEYikY8Hc3y_kLDYH5G?usp=sharing). Save them to folder `data/generated/` Note that this link is only accessible for people with USC email addresses.

Download the word embeddings from [this link](http://nlp.stanford.edu/data/glove.6B.zip) and extract the txt files to `methods/learning/models/word2vec/`

The resulting directory structure in the `data/` folder will look like the following:

```
./data/
├── aspect/
│   ├── annotations_voc_yelp_restaurants.txt
│   ├── aspects_restaurants.csv
│   └── lexicon_restaurants.csv
├── yelp_data/
│   ├── business.csv
│   ├── review.csv
│   └── user.csv
└── generated/
    ├── review.json
    ├── item.json
    └── user.json
```

## Running the Web Service

Run the web service at port 6010 by running the following command: `python3 components/app.py`.

## Current Project Status

### Feb 23rd, 2020

* We have completed user and item vector extraction based on reviews only.
* We have completed prototype for (user, item) score prediction model.
* We have finished the web service for score prediction; the API usage can be found at [this link](https://www.getpostman.com/collections/8973bb93151b84d82b38).

### Mar 8th, 2020

* We have completed the code for coping with cold start survey data.
* We have completed prototype for the recommendation API that gets user as input and outputs a list of business recommendations as output.
* We have updated the web service; the API usage can be found as the same link above.

### Mar 29th, 2020

* We have refactored the code for extending additional models for the recommendation engine.
* We have updated the aspect-based collaborative filtering recommendation engine with user review rating normalization. Now other user's ratings will be recorded as an overall mean over all reviews and a normalized score. This will help the engine deal with the bias of ratings given by different users.
* We have added evaluation code for the recommendation engine. Currently, we only support the RMSE error on the predicted item score. We will add additional metrics in the near future.

### Apr 12th, 2020

* We have refactored the codebase for supporting different methods while keeping the same components (datasets, etc.).
* We implemented a pipeline for training deep learning models with future extensions in mind.
* We implemented a MLP policy that predicts scores from user and item input pairs.
* For predicting visiting probability, we currently can directly map user scores to a probability value.
* We train our MLP model and provide results on the performance of this model. Training logs are available at [https://app.wandb.ai/jingyuny/faber](https://app.wandb.ai/jingyuny/faber).

### Apr 26th, 2020

* We have added the LSTM model that now can receive raw review texts for user as input and output score predictions (or visitating probability if we assume positive coordination between user score and visiting probability).
* We implemented the LSTM recommendation engine service.
* We have deployed the new service at the server we just set up for ML development [http://96.76.203.31:6010](http://96.76.203.31:6010).
