#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 18:06:23 2020

@author: ellenxiao
"""
import pandas as pd
import plotly.graph_objects as go

from Text_Preprocessing import preprocessing

df_covid1 = pd.read_pickle("df_covid1.pickle")

with open('vectoriser.pickle','rb') as f:
    pickle.load(vectoriser)

with open('model.pickle','rb') as f:
    pickle.load(RFmodel)
    

df_covid2 = df_covid1.drop_duplicates()

day_group = df_covid2.groupby(['create_date']).count()['text']
day_group = day_group.reset_index()

# plot
fig = go.Figure([go.Bar(x=day_group['create_date'], y=day_group['text'])])
fig.show()

import matplotlib.pyplot as plt
plt.bar(day_group['create_date'], day_group['text'], color='r')
plt.xlabel("create date")
plt.ylabel("number of tweets")
plt.title("number of tweets related to covid-19")
plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
plt.show()

clean_text = preprocessing(df_covid2['text'])
clean_text1 = vectoriser.transform(clean_text)

target1 = LRmodel.predict(clean_text1)

df_covid2['clean_text'] = clean_text
df_covid2['sentiment'] = target1

df_pos = df_covid2[df_covid2['sentiment']==1]
df_pos_group = pd.DataFrame(df_pos.groupby(['create_date']).count()['sentiment'])
df_pos_group.reset_index(inplace=True)
df_neg = df_covid2[df_covid2['sentiment']==0]
df_neg_group = pd.DataFrame(df_neg.groupby(['create_date']).count()['sentiment'])
df_neg_group.reset_index(inplace=True)

# group by sentiment and see how many tweets have been made for a day in the past week
fig = go.Figure() 
fig.add_trace(go.Scatter(x=df_covid2["date"], y=df_covid2['sentiment'], name= 'positive'))
fig.add_trace(go.Scatter(x=df_covid2["date"], y=df_covid2['sentiment'], name = 'negative'))
fig.update_layout(title='Sentiment towards COVID-19 on Twitter', xaxis_title='Date',yaxis_title='Number of Tweets')
fig.show()
# using matplotlib
plt.plot(df_pos_group['create_date'],df_pos_group['sentiment'])
plt.plot(df_neg_group['create_date'],df_neg_group['sentiment'])
plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
plt.xlabel("create date")
plt.ylabel("number of tweets")
plt.legend(['positive','negative'])
plt.title("number of tweets related to covid-19 in a week")
plt.show()