# Sentiment-Analysis-towards-Covid-19-on-Twitter

## Introduction
Coronavirus pandemic is definitely one of the most serious and disastrous things in 2020. It is a life changer for lots of people. In Massachusetts, US, people started working at home since March, staying at home order is still effective, grocery stores, supermarkets and many indoor business require customers to wear a mask, people keep social distance spontaneously. I believe many people, at least myself, felt panic, nervous at first, then felt bored staying at home, for now I kind of adapted to this new normal. 

I recently watched a Youtube video of [freeCodeCamp.org](https://www.youtube.com/watch?v=1gQ6uG5Ujiw), which inspires me a lot. It gives me a sense how to extract data from twitter API and do simple visualization and rule-based sentiment analysis using textBlob. However, rule-based algorithm is based on the text rather than content, which makes it less realistic. In order to increase the precision and accuracy, I will apply supervised machine learning classification algorithms and also unsupervised machine learning algorithms to explore data. 

This small project aims to find out how the people’s opinions and attitude changed overtime on twitter. The training dataset is being used is ‘sentiment140 dataset’, which contained extracted using the Twitter API and has been annotated positive/negative to detect sentiment. The dataset is pulled through Twitter API and are all about Covid-19 and Conoravirus. 

## Table of Content
- [Dataset](#Dataset) 
- [Data Preprocessing](#Data_Preprocessing)  
- [Train Model](#Train_Model)  
- [Get Titter Dataset](#Get_Titter_Dataset)  
- [Apply Model](#Apply_Model)

## Dataset
The data source can be found at [Kaggle](https://www.kaggle.com/kazanova/notebook) - Sentiment140 dataset with 1.6 million tweets. 

## Data_Preprocessing
<a href="https://github.com/ellenxxiao/Sentiment-Analysis-towards-Covid-19-on-Twitter/blob/master/Text_Preprocessing.py" target="_blank">Data preprocessing</a> contains several steps to clean the tweets text:
1. remove URL and username in the tweets 
2. remove consecutive letters (e.g., soo -> so)
3. convert emojis to words
4. remove non-alphabets 
5. lemmatization given the part of speech

## Train_Model
<a href="https://github.com/ellenxxiao/Sentiment-Analysis-towards-Covid-19-on-Twitter/blob/master/train_model.py" target="_blank">Train Model</a> section shows the followings:
1. split dataset to train and test datasets 
2. apply TF-IDF vectorizer   
3. apply and evaluate models (models include multinomial naive bayes classifier, linear SVM, Logistic Regression, and RandomForest Classifier)   

## Get_Titter_Dataset
Call twitter API to get recent 7 days tweets related to Covid-19, refer <a href="https://github.com/ellenxxiao/Sentiment-Analysis-towards-Covid-19-on-Twitter/blob/master/twitter_search2.py" target="_blank">here</a>

## Apply_Model
The score of four models are:     
- Multinomial naive bayes classifier - 0.776  
- Linear SVM - 0.777    
- Logistic Regression - 0.791     
- RandomForest Classifier - 0.753  
     
Therefore, selected the model with the highest score (Logistic Regression) for sentiment analysis. Code is <a href="https://github.com/ellenxxiao/Sentiment-Analysis-towards-Covid-19-on-Twitter/blob/master/apply_model.py" target="_blank">here</a>      
Let's see how many tweets about covid-19 have been made each day during the last week and how number of positive and negative posts changed in a week.     
![Capture](https://user-images.githubusercontent.com/26680796/90534540-dc1c6f00-e147-11ea-86a8-888136f60137.png)
![Capture](https://user-images.githubusercontent.com/26680796/90534562-e2aae680-e147-11ea-89e6-1cd948b930d5.png)



