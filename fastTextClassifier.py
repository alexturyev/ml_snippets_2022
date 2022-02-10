# works - fast text classifier. Not that great imo...
import re
import fasttext

FILE_DATASET = 'data/ft_train_dataset.txt'

def predictThis(model, utterance):
    rslt = model.predict(utterance.upper(), k=3)
    print(utterance, ' -> ', rslt)
    
print('Training...')
model = fasttext.train_supervised(FILE_DATASET, verbose=True, epoch=15)
print('Done training...')
print(model.labels)

predictThis(model, 'operator')
predictThis(model, 'representative')
predictThis(model, 'bill pay')
predictThis(model, 'I wanna talk')
predictThis(model, 'where is my money')
predictThis(model, 'change my address')
