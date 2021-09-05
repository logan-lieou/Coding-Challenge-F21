"""
THIS IS SLIGHTLY MODIFIED VERSION OF SOMEONE ELSES CODE
SOURCE CODE IS LOCATED HERE: https://www.kaggle.com/grenadebrain/create-labelled-dataframe
"""

import pandas as pd

dictionary = pd.read_csv('./SST2-Data/SST2-Data/stanfordSentimentTreebank/stanfordSentimentTreebank/dictionary.txt', sep='|')
dictionary.columns = ['body_text', 'phrase ids']
labels = pd.read_csv('./SST2-Data/SST2-Data/stanfordSentimentTreebank/stanfordSentimentTreebank/sentiment_labels.txt', sep='|')

df = dictionary.merge(labels, how='left', on='phrase ids')

df.to_csv("data.csv")