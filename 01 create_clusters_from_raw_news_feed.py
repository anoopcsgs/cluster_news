# -*- coding: utf-8 -*-
#Create Clusters of news
#returns index of news similar to each news item in the corpus
#
import os
os.chdir("C:/Users/anoop/nlp/nlp")
from time import time
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.pyplot as plt
from nltk.stem.porter import  PorterStemmer
from scipy.spatial.distance import euclidean
from sklearn.decomposition import LatentDirichletAllocation

from pymongo import MongoClient
import datetime
from config import constants

# organization data
client = MongoClient(constants.MONGO_IP, constants.MONGO_PORT)
db = client.sampledb
collection = db.news



#------------modify this bit with mongo inserts
#read files from news db
import glob
load_news = glob.glob("clustering/data/*.txt")
news_master = []

#read only 20 articles for testing and store in a list
for doc in load_news[0:20]:
    temp = open(str(doc)).read()
    news_master.append(temp)
    

X = np.array(news_master)
df_text = pd.DataFrame(X)
df_text.columns=['news']

#------------modify this bit with mongo inserts

#define preprocessors and cleansers
def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    return text

porter = PorterStemmer()
def tokenizer_porter(text):
    return[porter.stem(word) for word in text.split()]

nltk.download('stopwords')
stop = stopwords.words('english')

#clean text and normalize for usage
df_text['news'] = df_text['news'].apply(preprocessor)
X = df_text.iloc[:,0].values
tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None, tokenizer=tokenizer_porter, stop_words=stop)
tok = tfidf.fit_transform(X)

#perform clustering to get elbow
distortions = []
silh = []
for i in range(2, 15, 1):
    km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=300, random_state=0)
    km.fit(tok)
    distortions.append(km.inertia_)
    silh.append(metrics.silhouette_score(tok, km.labels_, sample_size=1000))
    
#to see the output (commented for now)
#plt.plot(range(2, 15, 1), distortions, marker='o')
#plt.plot(range(2, 15, 1), silh, marker='o',color='green')

#score data and get clusters    
clust_num = 5 #based on elbow...will be automated later

km = KMeans(n_clusters=clust_num, init='k-means++', n_init=10, max_iter=300, tol=1e-04, random_state=0)
tok = tfidf.fit_transform(X)
y_km = km.fit_predict(tok)
df_text['cluster_number'] = np.array(y_km)
cluster_output = pd.DataFrame(df_text) #this stores cluster number for each document


#create list of top words per cluster (compare with unsupervised LDA output)

#move print option to external file
print("Top 10 terms per cluster:")
order_centroids = km.cluster_centers_.argsort()[:, ::-1]
terms = tfidf.get_feature_names()
for i in range(5):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind]),
    print
    

