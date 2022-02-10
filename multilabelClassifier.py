# works with Brian's datum files.
import re
import math

import nltk
from nltk import word_tokenize
from nltk.sentiment.util import mark_negation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import TransformerMixin
import joblib
import pickle
import pandas as pd
import numpy as np
import random
from nltk.corpus import stopwords
from sklearn.preprocessing import MultiLabelBinarizer

#nltk.download('punkt')
#nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

modelFilename = 'model/datumModel.sav' # this is where model gets saved
TEST_FILENAME = 'data/datums_2022-01-27_13_30_30.csv' # the file that's used for testing
TRAINING_FILENAMES = ['data/datum_train.csv'] # list of files to use for training

# list of categories
CATEGORIES_STR = 'Month the Day,Twenty oh One,Leading Qualifier,Already Left,Day of week,Leading DoW,DateMonth,DigitDigit,Today,Yesterday,Tomorrow,Now,Month Date Good,Month Date Year Good,NonOrdinal'


phraseRows = {}
trainDataDic = {}

# split categories into list
CATEGORIES = list(set(CATEGORIES_STR.split(',')))

# function to clean text
def clean_text(text):
    if not isinstance(text, str) or len(text) == 0:
        return ''
    text = re.sub(r'[^\w\s]','',text, re.UNICODE)
    text = text.lower()
    #text = [lemmatizer.lemmatize(token) for token in text.split(" ")]
    #text = [lemmatizer.lemmatize(token, "v") for token in text]
    text = [word for word in text if not word in stop_words]
    #text = " ".join(text)
    return text

# determines if category is clean
def clean_category(catValue):
    #print(catValue)
    if not isinstance(catValue, str) or len(catValue) == 0:
        return 0
    if catValue.lower().strip() == 'y':
        return 1
    return 0

# creates numpy train sets
def getTrainSets():
    trainX = []
    trainY = []
    for word in trainDataDic:
        thisY = []
        trainX.append(word)
        for index, category in enumerate(CATEGORIES):
            if category in trainDataDic[word] and trainDataDic[word][category] > 0:
                thisY.append(category)
        trainY.append(thisY)
    return np.array(trainX), np.array(trainY)


def analyze_dataset():
  global trainDataDic
  categoryData = {}

  missingCount = 0
  headerList = ['clean(verbiage)']
  for cat in CATEGORIES:
      categoryData[cat] = []
      headerList.append(cat)
  totalList = [headerList]
  for word in trainDataDic:
      catCount = 0
      if len(word) == 0:
          continue
      wordInfo = [word]
      for cat in CATEGORIES:
          if cat in trainDataDic[word] and trainDataDic[word][cat] > 0:
              catCount = catCount + 1
              categoryData[cat].append(word)
              wordInfo.append('Y')
          else:
              wordInfo.append('')
      if catCount == 0:
          missingCount = missingCount + 1
          print(word,' - #',phraseRows[word])
      totalList.append(wordInfo)

  print("Total Missing", missingCount)
  #print(totalList)

  # you can print some info about the dataset into the file
  #tldf = pd.DataFrame(totalList)
  #tldf.to_csv('data/totalDataset.csv', header=False, index=False)
  #np.savetxt('data/totalDataset.csv', np.array(totalList), delimiter=',')

  for cat in categoryData:
      print('   ', cat, len(categoryData[cat]))

def addToTrainData(df):
    global trainDataDic, phraseRows
    skippedCount = 0
    for index, row in df.iterrows():
        phrase = row['clean']
        if 'Match' in row and row['Match'].strip() == "1":
            skippedCount = skippedCount + 1])
        else:
            if phrase not in trainDataDic:
                trainDataDic[phrase] = {}
                phraseRows[phrase] = index

            for cat in CATEGORIES:
                if cat in row:
                    trainDataDic[phrase][cat] = row[cat]

    print("Total recs skipped, because they are matched already:", skippedCount)

def loadTrainData():
    print("Reading training data...")
    for path in TRAINING_FILENAMES:
        df = readTrainCsv(path)
        addToTrainData(df)

def readTrainCsv(path):
    col_list = CATEGORIES.copy()
    col_list.insert(0, "clean")
    df = pd.read_csv(path)
    df['clean'] = df['clean'].map(clean_text)
    df = df.drop_duplicates()
    for cat in CATEGORIES:
        if cat in df.columns:
            df[cat] = df[cat].map(clean_category)

    return df

def trainModel(train_X, train_y):
    print("Creating classifier...")
    unigram_bigram_clf = Pipeline([
                ('cv', CountVectorizer(analyzer="word",
                       ngram_range=(1, 2),
                       tokenizer=word_tokenize,
                       #tokenizer=lambda text: mark_negation(word_tokenize(text)),
                       )),
                ('clf', OneVsRestClassifier(LogisticRegression(solver='newton-cg'))),
            ])

    print("Training...")
    unigram_bigram_clf.fit(train_X, train_y)
    print("Saving model as ", modelFilename)
    pickle.dump(unigram_bigram_clf, open(modelFilename, 'wb'))
    return unigram_bigram_clf

def loadModel():
    print("Loading model...")
    return joblib.load(modelFilename)

def parsePrediction(prediction):
    global mlb
    predictionSet = {}
    for index, proba in enumerate(prediction[0]):
        if proba > 0.5:
            predictionSet[mlb.classes_[index]] = proba
    return predictionSet

def testAll(model, data):
    preds = {}
    for text in data:
        preds[text] = testModel(model, text)

    return preds

def testModel(model, text):
    if not isinstance(text, list):
        text = [text]
    prediction = model.predict_proba(text)
    predictionSet = parsePrediction(prediction)
    #print(text[0], predictionSet)
    return predictionSet

testRowNumberDic = {}

def load_test(filename):
  # shallow copy categories
  global testRowNumberDic
  col_list = ['clean', 'acc-stat-no-rej']
  df = pd.read_csv(filename, usecols=col_list)
  df['clean'] = df['clean'].map(clean_text)
  skippedCount = 0
  test_list = {}
  for index, row in df.iterrows():
      phrase = row['clean']
      if 'acc-stat-no-rej' in row and row['acc-stat-no-rej'].strip() == "CA-in":
          skippedCount = skippedCount + 1
          #print("Skipping", row['clean'])
      else:
          if phrase not in test_list:
              testRowNumberDic[phrase] = index
              test_list[phrase] = 1
          else:
              test_list[phrase] = test_list[phrase] + 1

  return test_list

def saveResults(results, filename):
    headerList = ['row_number','clean(verbiage)']
    for cat in CATEGORIES:
        headerList.append(cat)
    totalList = [headerList]
    for text in results:
        if len(text) < 1:
            continue
        thisInfo = []
        if text in testRowNumberDic:
            thisInfo.append(testRowNumberDic[text]+2) #one for header row, one for 0 index
        else:
            thisInfo.append('')
        thisInfo.append(text)
        for cat in CATEGORIES:
            if cat in results[text] and results[text][cat] > 0.5:
                thisInfo.append('Y')
            else:
                thisInfo.append('')
        totalList.append(thisInfo)
    tldf = pd.DataFrame(totalList)
    tldf.to_csv(filename, header=False, index=False)

# saves dataset into a fastText file
def createFastTextFile(textFilename):
    with open(textFilename, 'w') as the_file:
        for word in trainDataDic:
            if len(word) < 1:
                continue
            thisLine = ''
            for index, category in enumerate(CATEGORIES):
                if category in trainDataDic[word] and trainDataDic[word][category] > 0:
                    thisLine = thisLine + '__label__' + category
            if len(thisLine) < 1:
                thisLine = word.strip() + ' \r\n'
            else:
                thisLine = thisLine + ' ' + word.strip() + ' \r\n'

            the_file.write(thisLine)
    #__label__toxic__label__racist__label__insult Text of the comment here


# load train data
loadTrainData()
print(len(trainDataDic),"train recs")

# create fastText format file from the dataset
#createFastTextFile('data/fastTextTrain.txt')

#analyze_dataset()

# create datasets for sklearn
trainX, trainY = getTrainSets()

# initialize MultiLabelBinarizer
mlb = MultiLabelBinarizer()
mlb.fit([CATEGORIES])

# transform categories and look at the shape
trainY = mlb.transform(trainY)
print(trainX.shape)
print(trainY.shape)

# train and get model
model = trainModel(trainX, trainY)

# load test dataset
testData = load_test(TEST_FILENAME)
print("Test data loaded with ", len(testData), " rows...")

# test and get results
results = testAll(model, testData)

# save results
saveResults(results, 'data/test_results_service_options.csv')

#print(results)
#model = loadModel()
#testModel(model, ['january']);
#testModel(model, ['December fourth and the rest of the month of December']);
#testModel(model, ['December sixth twenty twenty one']);
#testModel(model, ['december third']);
