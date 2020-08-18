#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 10:19:52 2020

@author: ellenxiao
"""

import twitter

import twitter_credentials

import requests
import json

import os
from requests_oauthlib import OAuth1Session

def call_api(create_date):
    params = {"q":"conoravirus","until":create_date,"count":"100","lang":"en"}
    oauth = OAuth1Session(twitter_credentials.consumer_key,
                           client_secret=twitter_credentials.consumer_secret,
                           resource_owner_key=twitter_credentials.access_token,
                           resource_owner_secret=twitter_credentials.access_token_secret)
    response = oauth.get("https://api.twitter.com/1.1/search/tweets.json", params = params)
    json_data = json.loads(response.text)
    return json_data

def add_to_dataframe(json_data,df_covid1):
    for r in json_data['statuses']:
        try:
            userid = r['id']
            date = r['created_at']
            retweet = r['retweet_count']
            favorite = r['favorite_count']
            text = r['text']
        except:
            pass
        row_insert = [userid, date, retweet, favorite, text]
        df_covid1.loc[-1] = row_insert
        df_covid1.index = df_covid1.index+1
        df_covid1 = df_covid1.sort_index()
    return df_covid1

df_covid1 = pd.DataFrame({"id":[],"date":[],"retweet_count":[],"favorite_count":[],"text":[]})
create_date="2020-08-10"
json_data=call_api(create_date)
for create_date in ["2020-08-10","2020-08-11","2020-08-12","2020-08-13","2020-08-14","2020-08-15","2020-08-16","2020-08-17"]:
    json_data=call_api(create_date)
    df_covid1 = add_to_dataframe(json_data,df_covid1)

df_covid1.head()
df_covid1['month'] = df_covid1['date'].apply(lambda row: row[4:7])
df_covid1['day'] = df_covid1['date'].apply(lambda row: row[8:10])
df_covid1['year'] = df_covid1['date'].apply(lambda row: row[26:30])
df_covid1['create_date'] = df_covid1[['month', 'day', 'year']].agg('/'.join, axis=1)

#with open('df_covid1.pickle', 'wb') as f:
#    pickle.dump(df_covid1, f)

df_covid2 = df_covid1.drop_duplicates()

day_group = df_covid2.groupby(['create_date']).count()

if __name__ == "__main__":
    twitter_search2()