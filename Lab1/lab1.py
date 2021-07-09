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

