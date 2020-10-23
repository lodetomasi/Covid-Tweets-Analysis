# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 13:08:42 2020

@author: morel
"""

import json
import re
import glob
import gzip
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
#import heapq
import os
import pandas as pd 
import numpy as np
import random
import math
import matplotlib.pyplot as plt 
import nltk
import copy
import itertools
from scipy.sparse import csr_matrix
import time
import networkx as nx
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.cluster import AgglomerativeClustering
from collections import Counter 
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

def paa(window):
    timeseries = fill_series(top100k,window)   #Considering time windows...
    PAA = {}     #PAA  (we reduce dimension)
    
    for diz in timeseries:
        PAA[diz] = np.mean(np.array(list(timeseries[diz].values())).reshape(-1, 4), axis=1)    #Time series grain of 12h   (we took 8 timestamps per day)  [reshape(-1,8) if 24h]
    
    return PAA 

def sax_ca(PAA, sax):
    SAX = {}
    
    for diz in PAA:        #These take around 20 seconds each
        if not np.all(PAA[diz] == PAA[diz][0]):
            SAX[diz] = sax.fit_transform(PAA[diz].reshape(1,-1))
        
    regex = "a+b?bb?a+?a+b?bba*"    
    
    SAX_CA = {} #Collective Attention
    regex = "a+b?bb?a+?a+b?bba*"   
    
    for el in SAX:
        if (bool(re.search(regex, ''.join(list(SAX[el].flatten()))))):     #Check whether the regex appears in the time series
            SAX_CA[el] = SAX[el]
        
    return SAX_CA

def clustering(sax):
    df = pd.DataFrame(0, index = sorted(sax.keys()), columns = sorted(sax.keys())) 
    distances = []
    
    for el1 in sorted(sax):
        for el2 in sorted(sax):
            #if el2>el1:
                distances.append(np.count_nonzero(sax[el2]!=sax[el1]))
                
    lun = len(sax)
    
    for i in range (lun):
        df.iloc[i] = distances[i*lun:(i+1)*lun]
        
    n = round(lun/20)  #T'/20
    model = AgglomerativeClustering(affinity='precomputed', n_clusters=n, linkage='complete').fit(df)  #Using the clustering algorithm with the distance matrix just obtained 
    dict_labels = {k:v for k,v in zip(list(sax.keys()),model.labels_)}
    
    clusters = {}          #This will contain words per cluster
    for k, v in dict_labels.items():   #invert key-values of dictionary
        clusters.setdefault(v, []).append(k)
        
    return clusters

def sentence_groups(l):  #divide list of strings into list of lists of strings when there is a '-' (thus creating a list per tweet)
    group = []
    groups = []
    for w in l:
        group.append(w)
        if w == '-':
            group.remove(w)
            groups.append(group)
            group = []
    return groups    

def create_co_occurences_matrix(allowed_words, documents):
    word_to_id = dict(zip(allowed_words, range(len(allowed_words))))
    id_to_word = {v: k for k, v in word_to_id.items()}
    documents_as_ids = [np.sort([word_to_id[w] for w in doc if w in word_to_id]).astype('uint32') for doc in documents]
    row_ind, col_ind = zip(*itertools.chain(*[[(i, w) for w in doc] for i, doc in enumerate(documents_as_ids)]))
    data = np.ones(len(row_ind), dtype='uint32')  # use unsigned int for better memory utilization
    max_word_id = max(itertools.chain(*documents_as_ids)) + 1
    docs_words_matrix = csr_matrix((data, (row_ind, col_ind)), shape=(len(documents_as_ids), max_word_id))  # efficient arithmetic operations with CSR * CSR
    words_cooc_matrix = docs_words_matrix.T * docs_words_matrix  # multiplying docs_words_matrix with its transpose matrix would generate the co-occurences matrix
    words_cooc_matrix.setdiag(0)
    return words_cooc_matrix, id_to_word 

def co_occurrences_graph(window,cluster):
    #Looking for terms in documents in the given window
    allowed_words = clusters[window][cluster]  #The 'alphabet' whose co-occurrencies we're searching  (words contained in a certain cluster in a certain time window)
    words_cooc_matrix, word_to_id = create_co_occurences_matrix(allowed_words, words_per_tweet[window]) #These take around 30 seconds per cluster! (there are about 60 clusters per window --> 30 mins) 
    z = words_cooc_matrix.todense()
    
    #Creating the graph
    G = nx.from_numpy_matrix(z)
    G = nx.relabel_nodes(G, word_to_id) #Using the word as node label
    layout = nx.spring_layout(G)
    nx.draw(G, layout, with_labels=True)
    labels = nx.get_edge_attributes(G, "weight")
    pippo = nx.draw_networkx_edge_labels(G, pos=layout, edge_labels=labels)
    return G, pippo

def k_core(G,k):
    H=G.copy()
    i=1
    while (i>0):
        i=0
        for node in list(H.nodes()):
            if H.degree(node)<k:
                H.remove_node(node)
                i+=1
    if (H.order()!=0):
        plt.figure()
        plt.title(str(k) +'-core decomposition') 
        nx.draw(H,with_labels=True)
    return H

def full_k_core_decomposition(G):
    vuoto = False
    k=1
    while (vuoto==False):
        H = k_core(G,k)
        k+=1
        if (H.order()==0):
            vuoto = True
    
#%% 0.1 DOWNLOADING AND PROCESSING DATA
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
                g.write('\n'.join(text_preprocessing(tweetDict['retweeted_status']['full_text']))+'\n-\n')
            else:
                g.write('\n'.join(text_preprocessing(tweetDict['full_text']))+'\n-\n')               

        
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
        
    words.remove('-')

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

#%%0.2A BUILDING TIME SERIES        
                                              
PAA1 = paa(0)  #These will take about 1 minute in total
PAA2 = paa(1)
PAA3 = paa(2)
PAA4 = paa(3)
PAA5 = paa(4)
    
#%% 0.2B SAX
n_bins = 2
sax = SymbolicAggregateApproximation(n_bins=n_bins, strategy='uniform') #transform those time series into sequences of As and Bs
        
SAX1_CA = sax_ca(PAA1, sax)   #These will take about 20 seconds each
SAX2_CA = sax_ca(PAA2, sax)
SAX3_CA = sax_ca(PAA3, sax)
SAX4_CA = sax_ca(PAA4, sax)
SAX5_CA = sax_ca(PAA5, sax)
    
prova_ca = dict(random.sample(SAX3_CA.items(), 100))      

#%% 0.2C CLUSTERING    

clusters1 = clustering(SAX1_CA)  #These will take about 1/1.5 minutes each
clusters2 = clustering(SAX2_CA)
clusters3 = clustering(SAX3_CA)
clusters4 = clustering(SAX4_CA)
clusters5 = clustering(SAX5_CA)

clusters = [clusters1,clusters2,clusters3,clusters4,clusters5]

#%% 0.3 CO-OCCURRENCE GRAPH

#Generating documents
words_per_tweet = []
for i in range (5): #Cycling through each window 0,1,2,3,4  (Should take a couple of minutes)
    words2s = []   #We'll need 5 of these, or recycle it
    for filename in time_window(i):   #Cycling through tweets...
        with open(path + '/' + filename) as f:
            words2 = f.read().splitlines()
            [words2s.append(group) for group in sentence_groups(words2)]   #words2s now contains all tweets in a time_window    (This takes around 20 seconds per window)
    words_per_tweet.append(words2s)
    #for j in range (len (clusters[i])):   #It should continue here but it would take hours
        #G, grafo = co_occurrences_graph [i][j]

#Example of execution
G, grafo = co_occurrences_graph(0,2)

weights_dict = {key:int(grafo[key].get_text()) for key in grafo}
weights_dist = dict(Counter(weights_dict.values()).most_common()) #Weight distribution

full_k_core_decomposition(G) #Extracting K-Core

#%%  TESTS
#import heapq  #top k items of a dict
#pippo = heapq.nlargest(50, top100k, key=top100k.__getitem__)
#import random  #random sample of a dict
#prova = dict(random.sample(SAX1.items(), 50))   
            
