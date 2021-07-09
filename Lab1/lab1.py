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

# Declare a figure with a custom size
fig = plt.figure(figsize=(5, 5))
# labels for the classes
labels = 'ML-BSB-Lec', 'ML-HAP-Lec','ML-HAP-Lab'

# Sizes for each slide
sizes = [40, 35, 25]
# Declare pie chart, where the slices will be ordered and plotted counter-clockwise:
plt.pie(sizes, labels=labels, autopct='%.2f%%', shadow=True, startangle=90)
# autopct enables you to display the percent value using Python string formatting.
# For example, if autopct='%.2f', then for each pie wedge, the format string is '%.2f' and

# Equal aspect ratio ensures that pie is drawn as a circle.
plt.axis('equal')
# Display the chart
plt.show()

# Declare a figure with a custom size
fig = plt.figure(figsize=(5, 5))
# labels for the two classes
labels = 'Positives', 'Negative'
# Sizes for each slide
sizes = [len(all_positive_tweets), len(all_negative_tweets)]
# Declare pie chart, where the slices will be ordered and plotted counter-clockwise:
plt.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
# Equal aspect ratio ensures that pie is drawn as a circle.
plt.axis('equal')
# Display the chart
plt.show()

# printing positive tweets in greeen
print('\033[92m' + all_positive_tweets[random.randint(0,5000)])
# printing negative tweets in red
print('\033[91m' + all_negative_tweets[random.randint(0,5000)])

print()

# Our selected sample
tweet = all_positive_tweets[2277]
print(tweet)

# download the stopwords from NLTK
nltk.download('stopwords')

import re                                   # a library for the regular expression operations
import string                               # required for string operations
from nltk.corpus import stopwords           # a module for the stop words that come with NLTK lib
from nltk.stem import PorterStemmer         # a module for the stemming purpose
from nltk.tokenize import TweetTokenizer    # a module for tokenizing the strings

print('\033[92m' + tweet)
print('\033[94m')
# remove hyperlinks
tweet2 = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet2)
# remove hashtags
# only removing the hash # sign from the word
tweet2 = re.sub(r'#', '', tweet2)
print(tweet2)