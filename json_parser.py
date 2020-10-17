# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 13:08:42 2020

@author: morel
"""

import json
import re
import glob
import gzip
#import heapq
import os
import pandas as pd 
import numpy as np
import random
import math
import nltk
import copy
import time
import distance
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
#from nltk.tokenize import word_tokenize
#from pyts.approximation import PiecewiseAggregateApproximation
from pyts.approximation import SymbolicAggregateApproximation

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
    
#%% 0.1 DOWNLOADING DATA
#Opens the json files contained in the jsonl.gz archives, processes them and create some partial output txt files 

path = 'D:/Users/morel/Desktop/ing_inf/DATA_ANALYSIS/social_data_mining/project/COVID-19-TweetIDs-master/MarchTweets'

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

        
#%%   An example of a tweet processing
            
#print(text_preprocessing("@wadster13 @SaraCarterDC Question, does the attachment highlight an on going plan to distroy America by taking down President Trump and it's economy. Just because there is no shots fired don't mind that we're not at war https://t.co/MgEENN04ez"))

#%% 0.2 IDENTIFYING TOP 100k TOKENS
#Opens the txt files just generated, computes the TF-IDF of words and identifies the most common ones
                
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

#%%0.2A BUILDING TIME-SERIES        
                                             
timeseries1 = fill_series(top100k,0)         #Considering time windows...
timeseries2 = fill_series(top100k,1)
timeseries3 = fill_series(top100k,2)
timeseries4 = fill_series(top100k,3)
timeseries5 = fill_series(top100k,4)

PAA1 = {}          #PAA  (we reduce dimension)
PAA2 = {}
PAA3 = {}
PAA4 = {}
PAA5 = {}
for diz in timeseries1:
    PAA1[diz] = np.mean(np.array(list(timeseries1[diz].values())).reshape(-1, 4), axis=1)    #Time series grain of 12h   (we took 8 timestamps per day)
for diz in timeseries2:
    PAA2[diz] = np.mean(np.array(list(timeseries2[diz].values())).reshape(-1, 4), axis=1)
for diz in timeseries3:
    PAA3[diz] = np.mean(np.array(list(timeseries3[diz].values())).reshape(-1, 4), axis=1)
for diz in timeseries4:
    PAA4[diz] = np.mean(np.array(list(timeseries4[diz].values())).reshape(-1, 4), axis=1)
for diz in timeseries5:
    PAA5[diz] = np.mean(np.array(list(timeseries5[diz].values())).reshape(-1, 4), axis=1)
    
#%% 0.2B SAX
n_bins = 2
sax = SymbolicAggregateApproximation(n_bins=n_bins, strategy='uniform') #transform those time series into sequences of As and Bs

SAX1 = {}
SAX2 = {}
SAX3 = {}
SAX4 = {}
SAX5 = {}

for diz in PAA1:        #These take around 20 seconds each
    if not np.all(PAA1[diz] == PAA1[diz][0]):
        SAX1[diz] = sax.fit_transform(PAA1[diz].reshape(1,-1))
for diz in PAA2:
    if not np.all(PAA2[diz] == PAA2[diz][0]):
        SAX2[diz] = sax.fit_transform(PAA2[diz].reshape(1,-1))
for diz in PAA3:
    if not np.all(PAA3[diz] == PAA3[diz][0]):
        SAX3[diz] = sax.fit_transform(PAA3[diz].reshape(1,-1))
for diz in PAA4:     
    if not np.all(PAA4[diz] == PAA4[diz][0]):
        SAX4[diz] = sax.fit_transform(PAA4[diz].reshape(1,-1))
for diz in PAA5:
    if not np.all(PAA5[diz] == PAA5[diz][0]):
        SAX5[diz] = sax.fit_transform(PAA5[diz].reshape(1,-1))

#FILTERING FOR COLLECTIVE ATTENTION      
regex = "a+b?bb?a+?a+b?bba*"    

SAX1_CA = {}
SAX2_CA = {}
SAX3_CA = {}
SAX4_CA = {}
SAX5_CA = {}

for el in SAX1:
    if (bool(re.search(regex, ''.join(list(SAX1[el].flatten()))))):     #Check whether the regex appears in the time series
        SAX1_CA[el] = SAX1[el]

for el in SAX2:
    if (bool(re.search(regex, ''.join(list(SAX2[el].flatten()))))):
        SAX2_CA[el] = SAX2[el]

for el in SAX3:
    if (bool(re.search(regex, ''.join(list(SAX3[el].flatten()))))):
        SAX3_CA[el] = SAX3[el]

for el in SAX4:
    if (bool(re.search(regex, ''.join(list(SAX4[el].flatten()))))):
        SAX4_CA[el] = SAX4[el]

for el in SAX5:
    if (bool(re.search(regex, ''.join(list(SAX5[el].flatten()))))):
        SAX5_CA[el] = SAX5[el]
    
prova_ca = dict(random.sample(SAX3_CA.items(), 100))      

#%% 0.2C CLUSTERING    
#THESE LINES OF CODE SHOULD BE REPEATED FOR EACH TIME WINDOW... :)

df = pd.DataFrame(0, index = sorted(SAX5_CA.keys()), columns = sorted(SAX5_CA.keys())) 

distances=[]

for el1 in sorted(SAX5_CA):
    for el2 in sorted(SAX5_CA):
        #if el2>el1:
            distances.append(np.count_nonzero(SAX5_CA[el2]!=SAX5_CA[el1]))
            #df.loc[el1,el2] = np.count_nonzero(prova_ca[el2]!=prova_ca[el1])

lun = len(SAX5_CA)           
#df2 = pd.DataFrame(0, index = sorted(prova_ca.keys()), columns = sorted(prova_ca.keys())) 
for i in range (lun):
    df.iloc[i] = distances[i*lun:(i+1)*lun]

from sklearn.cluster import AgglomerativeClustering   #Using this clustering algorithm with the distance matrix just obtained 
n = round(lun/20)  #T'/20
model = AgglomerativeClustering(affinity='precomputed', n_clusters=n, linkage='complete').fit(df)
dict_labels = {k:v for k,v in zip(list(SAX5_CA.keys()),model.labels_)}

clusters = {}          #This will contain words per cluster
for k, v in dict_labels.items():
    clusters.setdefault(v, []).append(k)
       
#%%  TESTS
#import heapq  #top k items of a dict
#pippo = heapq.nlargest(50, timeseries3, key=top100k.__getitem__)

#import random  #random sample of a dict
#prova = dict(random.sample(SAX1.items(), 50))   

