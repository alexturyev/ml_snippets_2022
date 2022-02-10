# word embedding using glove or fasttext. OKish results
utterances = ['january','february','March','April','May','June','monday','wednesday','friday', 'weekend', 'pear', 'apple', 'cherry', 'tomato', 'pea', 'strawberry', 'pie', 'cake', 'ramen', 'noodles', 'mac and cheese']

from transformers import EncoderDecoderModel, BertTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np

corpus = []
for utt in utterances:
    corpus.append(utt.lower())


#embedder = SentenceTransformer('bert-base-nli-stsb-mean-tokens')



import torch
from models import InferSent

V = 1
MODEL_PATH = 'encoder/infersent%s.pkl' % V
params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': V}
model = InferSent(params_model)
model.load_state_dict(torch.load(MODEL_PATH))

W2V_PATH = 'model/crawl-300d-2M.vec'
W2V_PATH = 'model/glove.6B.300d.txt'
model.set_w2v_path(W2V_PATH)

model.build_vocab(corpus, tokenize=True)

corpus_embeddings = model.encode(corpus, tokenize=True)

from sklearn.cluster import KMeans
num_clusters = 4
# Define kmeans model
clustering_model = KMeans(n_clusters=num_clusters)
# Fit the embedding with kmeans clustering.
clustering_model.fit(np.array(corpus_embeddings, dtype=object))
# Get the cluster id assigned to each news headline.
cluster_assignment = clustering_model.labels_

def analyzeCluters(clusters):
    clusterDic = {}
    for index in range(len(clusters)):
        c = clusters[index]
        if c not in clusterDic:
            clusterDic[c] = []
        clusterDic[c].append(utterances[index])

    for c in clusterDic:
        print("---Cluster #", str(c))
        for sent in clusterDic[c]:
            print(sent)
        print('')

analyzeCluters(cluster_assignment)
