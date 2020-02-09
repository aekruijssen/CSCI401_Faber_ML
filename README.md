# CSCI401 Faber ML

## Setup

Download the Yelp dataset from [this link](https://www.kaggle.com/yelp-dataset/yelp-dataset/version/4). Save the CSV files `yelp_business.csv`, `yelp_review.csv` and `yelp_user.csv` to directory `data/` and rename to `business.csv`, `review.csv`, and `user.csv`.

Download the aspect annotation information [aspect_restaurants.csv](http://ir.ii.uam.es/aspects/data/vocabularies/aspects_restaurants.zip), [lexicon_restaurants.csv](http://ir.ii.uam.es/aspects/data/lexicons/lexicon_restaurants.zip), and [annotations_voc_yelp_restaurants.txt](http://ir.ii.uam.es/aspects/data/annotations/voc/annotations_voc_yelp_restaurants.zip). Extract the zip files and save the enclosed files to `aspect/`.

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
└── (other files and folders)
```
