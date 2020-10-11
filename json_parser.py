# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 13:08:42 2020

@author: morel
"""

import json
import glob
import gzip
import heapq
import os
import pandas as pd 
import numpy as np
import re
import math
import nltk
import copy
import time
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from pyts.approximation import PiecewiseAggregateApproximation

cachedStopWords = stopwords.words("english")   #this HUGELY improves exec time

#%%

def decontracted(phrase): #It helps the stemmer to recognize stuff
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

def text_preprocessing(input_text):
    input_text = re.sub("(RT )?@[A-Za-z0-9_]+","", input_text) #strips RTs and tags
    input_text = re.sub(r'http\S+', '', input_text)    #strips URLs
    input_text = input_text.lower()    #to LowerCase
    input_text = decontracted(input_text)     #expands contractions
    input_text = re.sub("[^a-zA-Z ]+", " ", input_text).strip()    #strips numbers, punctuation and leading/ending spaces
    tokenized = nltk.word_tokenize (input_text)    #tokenizes    
    tokenized = [i for i in tokenized if not i in cachedStopWords]  #remove stopwords
    stemmer= PorterStemmer()
    tokenized = [stemmer.stem(word) for word in tokenized]   #Porter stemming

    return tokenized

def chiavi():  #Returns a list of strings containing keys about day and hour (spans the entire month)
    chiavi = []
    rad = 'TweetsText-03-'
    for i in range(1,31):
        for j in range (1,25,3):
            chiavi.append(rad + str(i).zfill(2) + '-' + str(j).zfill(2) + '.txt')
    return chiavi

def time_window(n):  #Returns a list of strings containing keys about day and hour (spans the given window)
    if (n==0):
        return chiavi()[:80]
    else:
        return chiavi()[40*n:40*n+80]
    
def fill_series(diz,window):    #Takes the most common 100k terms and fills their timeseries with 0s in hours in which they are not used
    timeseries = nested_dict()
    window = time_window(window)
    for term in diz:
        #print('The normalized frequency of the term \'{}\' over time is: {}'.format(term,dict.__repr__(tf[term]).replace('TweetsText-','')))
        timeseries[term] = copy.deepcopy(tf[term])
        for key, value in list(timeseries[term].items()):
            if key not in window:
                del timeseries[term][key]
        for el in window:
            if el not in timeseries[term]:
                timeseries[term][el] = 0
    return timeseries
    
#%% Opens the json files contained in the jsonl.gz archives, processes them and create some partial output txt files 

path = 'D:/Users/morel/Desktop/ing_inf/DATA_ANALYSIS/social_data_mining/project/COVID-19-TweetIDs-master/MarchTweets'
#df = pd.DataFrame()
#start = time.time()

for filename in glob.glob(os.path.join(path, '*.jsonl.gz')):
    
    nome = 'TweetsText' + filename[-18:-9]    #It will be  '-MM-DD-HH'  (Month-Day-Hour)    
    print("Started creating {}".format(nome))
    
    with gzip.open(filename, 'rb') as f:
     if (nome + '.txt') not in os.listdir(path):   #Checks if this one hasn't already been processed
      with open(path + '//' + nome + '.txt', 'w') as g:
        for jsonObj in f:    #Each tweet in a file is a different json object
            tweetDict = json.loads(jsonObj)
            if "retweeted_status" in tweetDict:   #retweets and normal tweets have a DIFFERENT STRUCTURE! the former's "full_text" is truncated, so it's necessary to load the original tweet
                g.write('\n'.join(text_preprocessing(tweetDict['retweeted_status']['full_text'])))
            else:
                g.write('\n'.join(text_preprocessing(tweetDict['full_text'])))

#end=time.time()
#print(end-start)                             

#noret = []                             #Tweets that are NOT retweets are VERY FEW
#for text in globals()[nome]:
#    if not text.startswith('RT @'):
#        noret.append(text)
        
#%%   An example of a tweet processing
            
#print(text_preprocessing("@wadster13 @SaraCarterDC Question, does the attachment highlight an on going plan to distroy America by taking down President Trump and it's economy. Just because there is no shots fired don't mind that we're not at war https://t.co/MgEENN04ez"))

#%% Open the txt files just generated, computes the TF-IDF of words and identifies the most common ones
                
import collections 
path = 'D:/Users/morel/Desktop/ing_inf/DATA_ANALYSIS/social_data_mining/project/COVID-19-TweetIDs-master/MarchTweets'

def nested_dict():
    return collections.defaultdict(nested_dict)

max_freq={}            #Calculating TF-IDF
idf={}
tf=nested_dict()
docs = 0

for filename in os.listdir(path):
  if filename.endswith(".txt"): 
    print("Started processing {}".format(filename))
    docs+=1
    max_freq[filename] = 0
    
    freq = {}
    with open(path + '/' + filename) as f:
        words = f.read().splitlines()

    for word in words:                                                    
        if word not in freq:
            freq[word] = 0
            if word in idf:
                idf[word]+=1
            else:
                idf[word] = 1
        freq[word] += 1
        max_freq[filename] = max(max_freq[filename],freq[word])
                    
    for word in words:
       tf[word][filename]=freq[word]/max_freq[filename]
   
for a in idf:
    idf[a]=math.log(docs/idf[a],2)   
    
tf_idf=copy.deepcopy(tf)
sums={}
for a in tf_idf:
    for b in tf_idf[a]:
            tf_idf[a][b] = tf_idf[a][b]*idf[a]    
    sums[a] = sum(tf_idf[a].values()) 

n = 100000          #Only top 100k tokens are necessary 
top100k = dict(collections.Counter(sums).most_common(n))                                                            
timeseries1 = fill_series(top100k,0)         #Considering time windows...
timeseries2 = fill_series(top100k,1)
timeseries3 = fill_series(top100k,2)
timeseries4 = fill_series(top100k,3)
timeseries5 = fill_series(top100k,4)

PAA1 = {}
PAA2 = {}
PAA3 = {}
PAA4 = {}
PAA5 = {}
for diz in timeseries1:
    PAA1[diz] = np.mean(np.array(list(timeseries1[diz].values())).reshape(-1, 8), axis=1)    #Time series grain of 24h   (we took 8 timestamps per day)
for diz in timeseries2:
    PAA2[diz] = np.mean(np.array(list(timeseries1[diz].values())).reshape(-1, 8), axis=1)
for diz in timeseries3:
    PAA3[diz] = np.mean(np.array(list(timeseries1[diz].values())).reshape(-1, 8), axis=1)
for diz in timeseries4:
    PAA4[diz] = np.mean(np.array(list(timeseries1[diz].values())).reshape(-1, 8), axis=1)
for diz in timeseries5:
    PAA5[diz] = np.mean(np.array(list(timeseries1[diz].values())).reshape(-1, 8), axis=1)
       
#%%  TESTS
       
#import time
#start = time.time()
#for i in range (1000):
#    text_preprocessing(tweetDict['full_text'])
#end = time.time() 
#print(end-start)       
        
timeseries_flatted = pd.DataFrame.from_dict({(i,j): timeseries1[i][j] 
                           for i in timeseries1.keys() 
                           for j in timeseries1[i].keys()},
                       orient='index')

timetimeseries_flatted=timeseries_flatted.reset_index()

timetimeseries_flatted = timetimeseries_flatted.rename(columns = {'index':'filename'})
timetimeseries_flatted = timetimeseries_flatted.rename(columns = {0:'value'})
timetimeseries_flatted[['word','timestamp']] = timetimeseries_flatted.filename.str.split(",",expand=True,)
