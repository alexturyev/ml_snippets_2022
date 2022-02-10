# works great, but distilbertSimple is easier to use
import random
import mysql.connector
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import torch
import transformers as ppb
import warnings
import time

warnings.filterwarnings('ignore')

sents = ['operator', 'can i speak to an agent', 'check my balance']
mydb = mysql.connector.connect(
  host="127.0.0.1",
  user="root",
  password="we8b4n00n",
  database="trans"
)

def createDataset():
    cursor = mydb.cursor()
    ds = []
    labels = []
    texts = []
    query = ("SELECT robust_swbd_trans, google_intent FROM recs WHERE google_intent is not null and robust_swbd_trans != ''")

    cursor.execute(query)

    for (robust_swbd_trans, google_intent) in cursor:
        if len(google_intent) < 1:
            google_intent = 'Unknown'
        labels.append(google_intent)
        texts.append(robust_swbd_trans.lower())
        ds.append({
            'sentence': robust_swbd_trans.lower(),
            'label': google_intent
        })

    cursor.close()
    mydb.close()
    return texts, labels

def tokenizeSents(sentArr, maxLen = 0):
    arr = []
    for sent in sentArr:
        arr.append(tokenizer.encode(sent, add_special_tokens=True))
    if maxLen == 0:
        for i in arr:
            if len(i) > maxLen:
                maxLen = len(i)
    padded = np.array([i + [0]*(maxLen-len(i)) for i in arr])
    return padded, maxLen

trainX, trainY = createDataset()

model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)

#trainX = pd.DataFrame(trainX)
padded, max_len = tokenizeSents(trainX)

print(np.array(padded).shape)

attention_mask = np.where(padded != 0, 1, 0)

input_ids = torch.tensor(padded).long()
attention_mask = torch.tensor(attention_mask).long()

with torch.no_grad():
    last_hidden_states = model(input_ids, attention_mask=attention_mask)

features = last_hidden_states[0][:,0,:].numpy()
print('Features')
print(features.shape)
tokSents, max_len = tokenizeSents(sents, max_len)
print('TokSents')
print(tokSents.shape)
lr_clf = LogisticRegression()
lr_clf.fit(features, trainY)

rslt = lr_clf.score(features, trainY)
print(rslt)
start_time = time.time()
input_ids_test = torch.tensor(tokSents).long()
attention_mask_test = np.where(tokSents != 0, 1, 0)

input_ids = torch.tensor(padded).long()
attention_mask_test = torch.tensor(attention_mask_test).long()
with torch.no_grad():
    last_hidden_states_test = model(input_ids_test, attention_mask=attention_mask_test)

features_test = last_hidden_states_test[0][:,0,:].numpy()
mid_time = time.time()
preds = lr_clf.predict(features_test)
end_time = time.time()
print(str(mid_time - start_time), str(end_time - mid_time))
print(preds)
