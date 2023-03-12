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

The Colab Nootbook can be downloaded here: [Spam Detection](https://colab.research.google.com/drive/17my6zWZR8hrFyJJmSfxNe1GUBkizrVsX?usp=sharing).

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
| Actual Negative   | False Positive (FP) | True Negative (TN)  |  

   The entries in the diagonal (TP and TN) represent correct predictions, while the off-diagonal entries (FP and FN) represent incorrect predictions.       
- Precision Recall Curve: It plots the precision and recall scores of a model for different classification thresholds.

## Results

This is the graphical overview of the classifications performnance:

### Naive Bayes

  <img width="478" alt="Screenshot 2023-03-12 at 15 19 00" src="https://user-images.githubusercontent.com/54292420/224571172-4f7fb45d-8af9-488d-bfd6-5e3d902d4b3f.png"> <br> <img width="450" alt="Screenshot 2023-03-12 at 15 26 42" src="https://user-images.githubusercontent.com/54292420/224571627-8cceb562-dac5-4c8a-b1fd-bc4b06a812ef.png"> <br> <img width="434" alt="Screenshot 2023-03-12 at 15 28 35" src="https://user-images.githubusercontent.com/54292420/224571692-be7c5494-d2f1-4cc7-a7c1-09ab77ce9dee.png">
  
### Decision Tree

  <img width="506" alt="Screenshot 2023-03-12 at 15 22 18" src="https://user-images.githubusercontent.com/54292420/224571364-6962b72c-5fcc-49b9-8407-4ed50672b8d8.png"> <br> <img width="450" alt="Screenshot 2023-03-12 at 15 27 08" src="https://user-images.githubusercontent.com/54292420/224571708-a654ec88-ff48-419e-8910-86d587315148.png"> <br> <img width="434" alt="Screenshot 2023-03-12 at 15 28 48" src="https://user-images.githubusercontent.com/54292420/224571713-84a70f79-710f-40f2-bba4-78c70d6e3478.png">
  
### K-Nearest Neighbors

  <img width="493" alt="Screenshot 2023-03-12 at 15 24 48" src="https://user-images.githubusercontent.com/54292420/224571488-87bb141f-14b2-4d59-a66b-d37152a4807e.png"> <br> <img width="450" alt="Screenshot 2023-03-12 at 15 27 22" src="https://user-images.githubusercontent.com/54292420/224571740-5b583a1a-9e0e-48ee-9161-a5c2e94c53f8.png"> <br> <img width="434" alt="Screenshot 2023-03-12 at 15 28 59" src="https://user-images.githubusercontent.com/54292420/224571743-f4102574-e025-42bc-bb58-0d6a95a079b3.png">


