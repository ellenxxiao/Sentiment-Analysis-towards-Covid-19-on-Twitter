#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 11:16:20 2020

@author: ellenxiao
"""
import pickle 

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report

def Evaluation(model):
    y_pred = model.predict(X_test)
    
    score = model.score(X_test, y_test)
    print('Accuracy: ', score)
    '''
    cm = metrics.confusion_matrix(y_test, y_pred)
    f1_score = metrics.f1_score(y_test,y_pred)
    print('F1-Score:', f1_score)
    recall = (cm[1][1])/(cm[1][0]+cm[1][1])
    print('Sensitivity:', recall)
    
    categories = ['Negative','Positive']
    plt.figure(figsize=(7,7))
    sns.heatmap(cm, annot=True, fmt=".2f", linewidths=.5, square = True, 
                cmap = 'Blues_r',xticklabels=categories, yticklabels=categories);
    plt.ylabel('Actual label');
    plt.xlabel('Predicted label');
    all_sample_title = 'Accuracy Score: {0}'.format(score)
    '''
    
# import pickle as dataframe
df_text = pd.read_pickle('mypickle.pickle')
df_target = pd.read_pickle('mypickle_target.pickle')

df_target = df_target.replace(4,1)

X_train, X_test, y_train, y_test = train_test_split(df_text, target,test_size = 0.1, random_state = 0)

# TF-IDF Vectoriser
vectoriser = TfidfVectorizer(ngram_range=(1,2), max_features=500000)
vectoriser.fit(X_train)
print(f'Vectoriser fitted.')
print('No. of feature_words: ', len(vectoriser.get_feature_names()))

X_train = vectoriser.transform(X_train)
X_test  = vectoriser.transform(X_test)

y_train = y_train.replace(4,1)
y_test = y_test.replace(4,1)

# Multinomial Naive Bayes Classifier
NBmodel = MultinomialNB()
NBmodel.fit(X_train, y_train)
Evaluation(NBmodel) #0.77609

# Linear SVM
LSVCmodel = LinearSVC()
LSVCmodel.fit(X_train, y_train)
Evaluation(LSVCmodel) #0.77704

# Logistic Regression
LRmodel = LogisticRegression(C = 2, max_iter = 100, n_jobs=-1)
LRmodel.fit(X_train, y_train)
Evaluation(LRmodel) #0.79071

# Random Forest Classifier
RFmodel = RandomForestClassifier(n_estimators=100)
RFmodel.fit(X_train, y_train)
Evaluation(RFmodel) #0.75364

with open('vectoriser.pickle','wb') as f:
    pickle.dump(vectoriser, f)
    
with open('model.pickle','wb') as f:
    pickle.dump(LRmodel, f)