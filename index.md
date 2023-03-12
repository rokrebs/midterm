---
title: Spam Detection Using Machine Learning
author: Rocio Krebs
---


## Development Enviroment

For the Spam detection project, I decided to use Google Colab, which runs virtually and is accessible from any browser. It only requires a login with a google account to run the Notebook

## Libraries

The following libraries has been used: 
- pandas
- sklearn
- numpy
- nltk
- re
- seaborn
- matplotlib.pyplot

## Colab Notebook

The Colab Nootbook can be downloaded here: [Spam Detection] (https://colab.research.google.com/drive/17my6zWZR8hrFyJJmSfxNe1GUBkizrVsX?usp=sharing).

## Dataset Exploration

For Spam Detection, I used a Kaggle Dataset containing an extensive list of emails. First, I reviewed the dataset, printed out the dimension and an overview of how it looks and found out that the distribution of the emails needed to meet the requirements for my classification. It needed to include more spam emails. So I had to do a resampling of the dataset to achieve an equal class distribution, which means I got the same amount of spam and ham emails.


## Preprocessing

Before I could start implementing any classification algorithm. I had to prepare the emails in the right format, called preprocessing.
For preprocessing the emails.

I removed:

- Stop words 
- Numbers
- Symbols
- anything that was not a letter

Finally after I cleand the text of irrelevant information, I converted the text in marix of numbers.

## Classification

After the Preprocessing was done, I could start using the sckit-learn library with the following algorithms:

- Naive Bayes: A still very popular classifier for text classification is the naïve Bayes classifier, which gained popularity in applications of email spam filtering. Naïve Bayes classifiers are easy to implement, computationally efficient, and tend to perform particularly well on rela- tively small datasets compared to other algorithms.
- Decision Trees: this model as breaking down our data by making a decision based on asking a series of questions. Decision trees can build complex decision boundaries by dividing the feature space into rectangles. However, we have to be careful since the deeper the decision tree, the more complex the decision boundary becomes, which can easily result in overfitting.
- K-Nearest Neighbors: KNN is a typical example of a lazy learner. It is called “lazy” not because of its apparent simplicity, but because it doesn’t learn a discriminative function from the training data but memorizes the training dataset instead. 

Training using K-Nearest Neighbors takes long time running on the Colab Notebook.

## Performance

To evaluate the performance of the classifications algorithms, I used the following metrics:

- Precision: It measures the proportion of correctly classified positive instances out of all instances that the model predicted as positive.
- Recall: It measures the proportion of actual positive instances that were correctly classified by the model.
- f1-score: It is a metric used to evaluate the performance of a classification model that combines precision and recall. It is the harmonic mean of precision and recall and provides a single score that reflects both metrics.
- Confusius Matrix: It is a table used to evaluate the performance of a classification model. A confusion matrix consists of four entries: true positives (TP), true negatives (TN), false positives (FP), and false negatives (FN).<br>
|                   | Predicte Positive   | Predicte Negative   |
| :---------------- | :-----------------: | :-----------------: |
| Actual Positive   | True Positive (TP)  | False Negative (FN) | 
| Actual Negative   | False Positive (FP) | True Negative (TN)  |  <br>
The entries in the diagonal (TP and TN) represent correct predictions, while the off-diagonal entries (FP and FN) represent incorrect predictions.       
- Precision Recall Curve

## Results

This is the graphical overview of the classifications performnance:

- Naive Bayes
- Decision Tree:
- K-Nearest Neighbors


