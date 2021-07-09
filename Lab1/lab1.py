import nltk                                 # a Python library for NLP
from nltk.corpus import twitter_samples     # a sample Twitter dataset from NLTK
import matplotlib.pyplot as plt             # a library for visualization purpose
import random                               # a pseudo-random number generator

#downloads sample twitter dataset.
nltk.download('twitter_samples')

# select the set of positive and negative tweets
all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')

print('Number of positive tweets: ', len(all_positive_tweets))
print('Number of negative tweets: ', len(all_negative_tweets))
print('\nThe type of all_positive_tweets is: ', type(all_positive_tweets))
print('The type of a tweet entry is: ', type(all_negative_tweets[0]))

