# NLP_tweets

This repository contains all the files, scripts and notebooks used for the EPFL Machine Learning project about tweets sentiments classification.

## Data preparation
We can explore and prepare the data files to be read by the scripts using 'explore_data.py' and 'concat_csv.py'.
The first one shows a graph about the repartition of the words in the corpus, and cuts sentences by following the Zipf law.
The second one adds some preprocessing to the sentences (eg: removes multiple spaces, removes duplicates, replaces numbers, ...)
A word correction using jaccard_distance from nltk library could be used but needs too much computation for the size of our dataset.

## Models
'baseline' uses some NLP proven methods with n-grams probabilities, achieving an accuracy of ~82%.
'sentiment_analysis.ipynb' uses an LSTM model, achieving an accuracy of ~84%.
'transformer_V1.ipynb' uses a transformer model, achieving an accuracy of ~89%.
