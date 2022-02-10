# Alex ML Playground

This is my small list of quick working snippets of various Machine Learning approaches that I tested in early 2022 to get up to speed on the latest SotA developments in Sound Recognition and Text categorization.

## Requirements
> listing only major requirements. The code may run on lower versions of these

- Python 3.8.5
- scikit-learn 0.23.2
- nltk 3.6.7
- torch 1.6.0+cpu
- fasttext 0.9.2
- transformers 4.15.0
- GPU not required. Most of my testing was done with CPU since all of my testing is inference

## How to run
Most of this code requires some data from files or from DB. Unfortunately I cannot provide these, because it contains real production data. The code is really simple, so you can go through it and modify it to suit your needs. If you have any questions - feel free to contact me by raising an issue.

## ML Tasks

#### [distilbertSimple.py](https://github.com/alexturyev/ml_snippets_2022/main/distilbertSimple.py)
###### Task:
Large list of categorized sentences from around 50 categories from a database. Need to train a model to help categorize new sentences.

###### Method:
First using label encoder, then tokenizing with Bert and finally using a basic Logistic Regression to train.

###### Result:
I got every good results with this one. Training took a while, but its pretty awesome.

#### [fastTextClassifier.py](https://github.com/alexturyev/ml_snippets_2022/main/fastTextClassifier.py)
###### Task:
Large list of categorized sentences from around 50 categories from a file. Need to train a model to help categorize new sentences.

###### Method:
Using the whole FastText approach to train and predict.

###### Result:
Not so great. It kinda worked, but not sure if requires more tweaking. I could spend more time on this...

#### [multilabelClassifier.py](https://github.com/alexturyev/ml_snippets_2022/main/multilabelClassifier.py)
###### Task:
Text sentences are assigned multiple categories. Need to train with a fairly small dataset (20-50 per category) and test multi label (non exclusive) categorization.

###### Method:
Using MultiLabelBinarizer to feed categories into a very LogisticRegression model (using CountVectorizer to get basic vectors)

###### Result:
I didn't get amazing results, but I was surprised by how accurate it was with such a basic approach. It would be cool to compare different vectorizers (glove, fasttext, word2vec) or even bert.


#### [waveTranscribe.py](https://github.com/alexturyev/ml_snippets_2022/main/waveTranscribe.py)
###### Task:
List of audio files need to be transcribed using the latest SotA method

###### Method:
Using differently tuned version of Wav2Vec2.

###### Result:
Great results on nice data. Not so good results on real production data... probably needs to be tuned on a lot of transcribed calls. The best model in my testing was 'jonatasgrosman/wav2vec2-large-xlsr-53-english' because it was partially tuned on telephony data.


#### [clusterBestClasses.py](https://github.com/alexturyev/ml_snippets_2022/clusterBestClasses.py)
###### Task:
Lots of phrases and small sentences that need to be categorized into clusters (classes). Typical unsupervised clustering problem.

###### Method:
Using BertTokenizer and sklean's KMeans to do clustering. Best number of categories is determined using the Elbow method.

###### Result:
Really surprised with how accurate it was. I later used the same method on real production data and it was surprisingly good.

#### [zeroShot.py](https://github.com/alexturyev/ml_snippets_2022/zeroShot.py)
###### Task:
Lots of uncategorized text sentences need to be put into predefined categories. Classic Zero Shot classification problem.

###### Method:
Using "facebook/bart-large-mnli" to do zero shot classification. Given three categories, I am asking Bart to tell me which category a sentence belongs to.

###### Result:
I tried different sentences and didn't get the results I was hoping for. However, it might be useful if categories are very different from each other.
