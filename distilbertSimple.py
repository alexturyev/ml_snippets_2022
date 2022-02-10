# works great!!! similar to bertSimple, but easier to use
import numpy as np
import pandas as pd
import configparser
import torch
import transformers as ppb # pytorch transformers
import mysql.connector

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB
import warnings

config = configparser.ConfigParser()
config.read('config.ini')

sents = ['operator', 'can i speak to an agent', 'check my balance']

mydb = mysql.connector.connect(
  host = config['db']['host'],
  database = config['db']['database'],
  user = config['db']['user'],
  password = config['db']['password']
)

def createDataset():
    cursor = mydb.cursor()
    ds = []
    labels = []
    texts = []
    query = ("SELECT transcription, category_intent FROM recs WHERE category_intent is not null and transcription != ''")

    cursor.execute(query)

    for (transcription, category_intent) in cursor:
        if len(category_intent) < 1:
            category_intent = 'Unknown'
        labels.append(category_intent)
        texts.append(transcription.lower())
        ds.append({
            'sentence': transcription.lower(),
            'label': category_intent
        })

    cursor.close()
    mydb.close()
    return texts, labels

class BertTokenizer(object):

    def __init__(self, text=[]):
        self.text = text

        # Distilbert
        self.model_class, self.tokenizer_class, self.pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')

        # Bert
        #self.model_class, self.tokenizer_class, self.pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')

        # Load pretrained model/tokenizer
        self.tokenizer = self.tokenizer_class.from_pretrained(self.pretrained_weights)

        self.model = self.model_class.from_pretrained(self.pretrained_weights)

    def get(self):
        df = pd.DataFrame(data={"text":self.text})
        tokenized = []
        for sent in df["text"]:
            #print(sent)
            tokenized.append(self.tokenizer.encode(sent, add_special_tokens=True))
        #tokenized = df["text"].swifter.apply((lambda x: self.tokenizer.encode(x, add_special_tokens=True)))



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

trainX, trainY = createDataset()

print('Creating encoding labels...')
encoder = LabelEncoder()
trainY = encoder.fit_transform(trainY)

print('Creating Bert Tokenizer...')
_instance = BertTokenizer(text=trainX)
tokens = _instance.get()

print('Training LogisticRegression...')
lr_clf = LogisticRegression()
lr_clf.fit(tokens, trainY)

print('Creating Prediction Bert Tokenizer...')
_instance =BertTokenizer(text=sents)
tokensTest = _instance.get()

print('Predicting...')
predicted = lr_clf.predict(tokensTest)
print(predicted)
print(encoder.inverse_transform(predicted))
