#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 22:02:43 2020

@author: ellenxiao
"""
import re
import pandas as pd
import numpy as np
import plotly as ply
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords 
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk import pos_tag

import time
import pickle
    
class TweetLemmatizer():
    
    # get part of speech
    def get_wordnet_pos(self, tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return None
        
    def get_tweet_pos(self,sentence):
        global res
        
        stop_words = set(stopwords.words('english'))
        
        tokens = word_tokenize(sentence)
        tagged_sent = pos_tag(tokens)
        lemmatizer = WordNetLemmatizer()
        lemmas_sent = []
        for tag in tagged_sent:
            if tag[0] not in stop_words:
                wordnet_pos = self.get_wordnet_pos(tag[1]) or wordnet.NOUN
                lemmas_sent.append(lemmatizer.lemmatize(tag[0], pos=wordnet_pos)) # 词形还原
            res = " ".join(lemmas_sent)
        return res


# preprocessing text function
def preprocessing(text):
    
    processed = []
    
    #define emojis
    emojis = {':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad', 
              ':-(': 'sad', ':-<': 'sad', ':P': 'raspberry', ':O': 'surprised',
              ':-@': 'shocked', ':@': 'shocked',':-$': 'confused', ':\\': 'annoyed', 
              ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy',
              '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',
              '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile', ';)': 'wink', 
              ';-)': 'wink', 'O:-)': 'angel','O*-)': 'angel','(:-D': 'gossip', '=^.^=': 'cat'}
    
    urlPattern = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
    userPattern = '@[^\s]+'
    alphaPattern = "[^a-zA-Z0-9]"
    sequencePattern = r"(.)\1\1+"
    seqReplacePattern = r"\1\1"
    stop_words = set(stopwords.words('english')) 
    
    for tweet in text:
        tweet = tweet.lower()
        
        # Replace all URls with 'URL'
        tweet = re.sub(urlPattern,'',tweet)
        # Replace all emojis.
        for emoji in emojis.keys():
            tweet = tweet.replace(emoji, emojis[emoji])        
        # Replace @USERNAME to 'USER'.
        tweet = re.sub(userPattern,'', tweet)        
        # Replace all non alphabets.
        tweet = re.sub(alphaPattern, " ", tweet)
        # Replace 3 or more consecutive letters by 2 letter.
        tweet = re.sub(sequencePattern, seqReplacePattern, tweet)
        
    ##HERE HERE HERE
        lemmatize_tweet = TweetLemmatizer()
        tweetwords = lemmatize_tweet.get_tweet_pos(tweet)
    ##HERE HERE HERE
        processed.append(tweetwords)
    return processed


if __name__ == "__main__":

    file_name = "/Users/ellenxiao/Documents/Projects/Twitter Sentiment Analysis/Twitter Kaggle/training.1600000.processed.noemoticon.csv"
    df_emoti = pd.read_csv(file_name, encoding="ISO-8859-1",header=None)

    df_emoti.columns = ['target','ids','date','no_query','user','text']
    
    # plot text dataset
    sub_target = pd.DataFrame(df_emoti.groupby(['target'],as_index=False)['ids'].count())
    fig = go.Figure(go.Bar(x=sub_target['target'],y=sub_target['ids']))
    fig.update_layout(height=800, width=900, title_text="")
    fig.show()
    
    # drop no_query column
    df_emoti = df_emoti.drop(['no_query'],axis=1)
    target = df_emoti['target']
    
    t = time.time()
    text_col = df_emoti['text']
    df_text = preprocessing(list(text_col))
    print(f'Text Preprocessing complete.')
    print(f'Time Taken: {round(time.time()-t)} seconds')
    
    # plot Word-Cloud - Negative
    data_neg = df_text[:800000]
    wc = WordCloud(max_words = 1000 , width = 1600 , height = 800,
                  collocations=False).generate(" ".join(data_neg))
    plt.figure(figsize = (20,20))
    plt.imshow(wc)
    # plot Word-cloud - Positive
    data_pos = df_text[800000:]
    wc = WordCloud(max_words = 1000 , width = 1600 , height = 800,
                  collocations=False).generate(" ".join(data_pos))
    plt.figure(figsize = (20,20))
    plt.imshow(wc)
    
    # save a pickle
    with open('mypickle.pickle', 'wb') as f:
        pickle.dump(df_text, f)
    
    with open('mypickle_target.pickle','wb') as f:
        pickle.dump(target, f)
    

