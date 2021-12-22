# NLP_tweets

Challenge AICrowd EPFL ML Text Classification<br>
This repository contains all the files, scripts and notebooks used for this challenge, as part of our ML course (Project 2).

## 1. Objective

Given two datasets of tweets, containing happy or sad smiley faces, the goal is to accurately predict the sentiment of new tweets, based only on their textual content.

## 2. How to use "transformer.ipynb"?

Files needed: <br>
* twitter-datasets/test_data.txt
* twitter-datasets/train_neg.txt
* twitter-datasets/train_pos.txt
* transformer.ipynb

twitter-datasets/test_data.txt: file containing the tweets whose class we want to predict.<br>
twitter-datasets/train_neg.txt: file containing a (short) list of tweets that we know contained a sad emoji.<br>
twitter-datasets/train_pos.txt: file containing a (short) list of tweets that we know contained a happy emoji.<br>
transformer.ipynb: Jupyter Notebook containing the code to predict the sentiment of the tweets in "test_data.txt" based on the other two.

On the AICrowd challenge page, there also are two additional files: "train_neg_full.txt" and "train_pos_full.txt".<br>
Their size is too big to be accepted by Github, and they weren't used in our code, this is the reason why these files are not in this repository.

We recommend the use of Google Colab if you do not have a GPU on which you can train your model.

### Colab (recommended)

The first three need to be uploaded to a Google Drive folder (with the same account as the one you use on Colab). In the current script, the files are stored in a folder "data" on the Google Drive. <br>
The jupyter notebook can then be uploaded to Colab.<br>
It should then be ready to use.

### Local

For a local execution, you must comment/delete all the cells with the comment `# Specific to Colab`.<br>
You should then be ready to execute the code.

## 3. What is happening in "transformer.ipynb"?

### Libraries

Installation and import of the libraries needed to the script.<br>
We will in particular use the library `transformers` which will allow use to work with state-of-the-art Natural Language Processing techniques. It provides with numerous pre-trained models that can be applied on text, image or audio. Here, it is the text part that is interesting to us.

### Seed randomness

Allow for the process to be replicated without too much perturbation.

### Form the correct dataset

In order to use the transformers, we are first obligated to extract our data from the files and transform it into a dataset. We then convert the sentences in this dataset into sequences of numbers using the BERT model.<br>
We also split the resulting dataset into two (train: 0.75 / validation: 0.25) so as to have meaningful statistics during the training.<br>
Additionally, we separate the whole datasets into batches to facilitate the training.

### Neural Network

Here is where we are declaring the Transformer class. It is the core of the script. The class itself is relatively basic, the initialization of the weights is done with the aforementioned library `transformers`.

We then define the `train` and `evaluate` functions. Nothing fancy here either, though we keep track of the accuracy and loss for each batch.

For the training part, we had to restrain to 3 epochs because of the duration of the training (3h+ on Google Colab). This hasn't been a problem anyway, the validation accuracy not making noticeable progress after the second epoch.

The displays show the average loss (resp. average accuracy) for each batch for each iteration.<br>
Although nothing obvious seems to appear in the graph of the loss, the graph of the accuracy shows a clear improvement with regards to the train accuracy (three distinct steps) while the validation accuracy stagnates during the whole process.

### Predictions

Once the model has been trained comes the moment to make the predictions of the classes of each test tweet. It is what is performed here.

## 4. What is in Archives?

<!-- TODO: Reformulate from here-->
### Data preparation
We can explore and prepare the data files to be read by the scripts using 'explore_data.py' and 'concat_csv.py'.

The first one shows a graph about the repartition of the words in the corpus, and cuts sentences by following the Zipf law.

The second one adds some preprocessing to the sentences (eg: removes multiple spaces, removes duplicates, replaces numbers, ...)

A word correction using jaccard_distance from nltk library could be used but needs too much computation for the size of our dataset.

### Models
'baseline_ngramsweightedaverage.ipynb' uses some NLP proven methods with n-grams probabilities, achieving an accuracy of ~82%.

'sentiment_analysis.ipynb' uses an LSTM model, achieving an accuracy of ~84%.

'transformer_V1.ipynb' uses a transformer model, achieving an accuracy of ~89%.
<!-- TODO: To here -->