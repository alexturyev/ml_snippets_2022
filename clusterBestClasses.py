# unsupervised clustering!!!
import pandas as pd
import re
import numpy as np
import pandas as pd

import torch
import transformers as ppb # pytorch transformers

from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from kneed import KneeLocator # this library helps to figure out how many clusters should be used

import warnings

utterances = ['january','february','March','April','May','June','monday','wednesday','friday', 'weekend', 'pear', 'apple', 'cherry', 'tomato', 'peas', 'strawberry', 'pie', 'cake', 'ramen', 'noodles', 'mac and cheese']

corpus = []
for utt in utterances:
    corpus.append(utt.lower())

class BertTokenizer(object):

    def __init__(self, text=[]):
        self.text = text

        # For DistilBERT:
        self.model_class, self.tokenizer_class, self.pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')

        # Load pretrained model/tokenizer
        self.tokenizer = self.tokenizer_class.from_pretrained(self.pretrained_weights)

        self.model = self.model_class.from_pretrained(self.pretrained_weights)

    def get(self):

        df = pd.DataFrame(data={"text":self.text})
        tokenized = []
        for sent in df["text"]:
            #print(sent)
            tokenized.append(self.tokenizer.encode(sent, add_special_tokens=True))

        max_len = 0
        for i in tokenized:
            if len(i) > max_len:
                max_len = len(i)

        padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized])

        attention_mask = np.where(padded != 0, 1, 0)
        input_ids = torch.tensor(padded).long()
        attention_mask = torch.tensor(attention_mask).long()

        with torch.no_grad(): last_hidden_states = self.model(input_ids, attention_mask=attention_mask)

        features = last_hidden_states[0][:, 0, :].numpy()

        return features

# get optimal number of clusters using the knee point detection
def getOptimalClusters(tokens):
    distortions = []
    inertias = []
    mapping1 = {}
    mapping2 = {}
    K = range(1, 15) # may need to be modified for your needs. max clusters here is 15
    X = np.array(tokens, dtype=object)
    for k in K:
        # Building and fitting the model
        kmeanModel = KMeans(n_clusters=k)
        kmeanModel.fit(X)

        distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_,
                                            'euclidean'), axis=1)) / X.shape[0])
        inertias.append(kmeanModel.inertia_)

        mapping1[k] = sum(np.min(cdist(X, kmeanModel.cluster_centers_,
                                       'euclidean'), axis=1)) / X.shape[0]
        mapping2[k] = kmeanModel.inertia_

    y = distortions
    x = range(1, len(y)+1)

    kn = KneeLocator(x, y, curve='convex', direction='decreasing')

# print stats about clusters
def analyzeClusters(clusters):
    clusterDic = {}
    for index in range(len(clusters)):
        c = clusters[index]
        if c not in clusterDic:
            clusterDic[c] = []
        clusterDic[c].append(corpus[index])

    for c in clusterDic:
        print("---Cluster #", str(c))
        for sent in clusterDic[c]:
            print(sent)
        print('')


print('Creating Bert Tokenizer...')
_instance = BertTokenizer(text=corpus)
tokens = _instance.get()

num_clusters = getOptimalClusters(tokens)
#num_clusters = 13
# Define kmeans model
clustering_model = KMeans(n_clusters=num_clusters)
# Fit the embedding with kmeans clustering.
clustering_model.fit(X)
# Get the cluster id assigned to each news headline.
cluster_assignment = clustering_model.labels_

analyzeClusters(cluster_assignment)
