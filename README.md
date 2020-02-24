# CSCI401 Faber ML

## Setup

Create a virtual environment: `python3 -m virtualenv venv`.

Activate the environment: `source venv/bin/activate`.

Install packages: `pip3 install -r requirements.txt`.

Download the Yelp dataset from [this link](https://www.kaggle.com/yelp-dataset/yelp-dataset/version/4). Save the CSV files `yelp_business.csv`, `yelp_review.csv` and `yelp_user.csv` to directory `data/` and rename to `business.csv`, `review.csv`, and `user.csv`.

Download the aspect annotation information [aspect_restaurants.csv](http://ir.ii.uam.es/aspects/data/vocabularies/aspects_restaurants.zip), [lexicon_restaurants.csv](http://ir.ii.uam.es/aspects/data/lexicons/lexicon_restaurants.zip), and [annotations_voc_yelp_restaurants.txt](http://ir.ii.uam.es/aspects/data/annotations/voc/annotations_voc_yelp_restaurants.zip). Extract the zip files and save the enclosed files to `aspect/`.

Download the saved user and item vectors at [this link](https://drive.google.com/drive/folders/1Jt3U2ix-zsZljOEYikY8Hc3y_kLDYH5G?usp=sharing). Note that this link is only accessible for people with USC email addresses.

The resulting directory structure will look like the following:

```
/ (root directory for project)
├── aspect/
│   ├── annotations_voc_yelp_restaurants.txt
│   ├── aspects_restaurants.csv
│   └── lexicon_restaurants.csv
├── data/
│   ├── business.csv
│   ├── review.csv
│   └── user.csv
├── saved/
│   ├── item.json
│   └── user.json
└── (other files and folders)
```

## Running the Web Service

Run the web service at port 6010 by running the following command: `python3 app.py`.

## Current Project Status

* We have completed user and item vector extraction based on reviews only.
* We have completed prototype for (user, item) score prediction model.
* We have finished the web service for score prediction; the API usage can be found at [this link](https://www.getpostman.com/collections/8973bb93151b84d82b38).
