# works - transcribes large volumes of wav files using wav2vec2 model and inserts results into DB
import torch
import librosa
import numpy as np
import soundfile as sf
import time
import os
import re
import io
from IPython.display import Audio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import pandas as pd

# col_prefix = 'jsg_xlsr_trans' # database column name to save
RECS_REPO = 'recs/22/' # directory containing files
MODEL_NAME = "jonatasgrosman/wav2vec2-large-xlsr-53-english" #"./ml_server/venv/u-wav2vec2-large-robust-ft-swbd-300h" #"facebook/wav2vec2-large-robust-ft-swbd-300h" #"facebook/wav2vec2-large-960h-lv60-self" #"jonatasgrosman/wav2vec2-large-xlsr-53-english" # "facebook/wav2vec2-base-960h"

# transcribes a wav filename by loading it with librosa tokenizing and predicting using wav2vec2 tuned model
def transcribeThis(filename):
    start_time = time.time()
    input_audio, _ = librosa.load(filename, sr=16000)
    #print(input_audio)
    #input_audio, _ = librosa.load(filename, sr=16000)
    input_values = tokenizer(input_audio, return_tensors="pt").input_values
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = tokenizer.batch_decode(predicted_ids)[0]
    diffTime = time.time() - start_time
    print(transcription,diffTime)
    return transcription,diffTime

# returns list of wav files from a directory
def findFiles(dir):
    f = []
    for (dirpath, dirnames, filenames) in os.walk(dir):
        for file in filenames:
            x = re.search("\.wav$", file)
            if x:
                f.append(file)
    return f

# initialize tokenizer and model
tokenizer = Wav2Vec2Tokenizer.from_pretrained(MODEL_NAME)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)

# find recordings
file_names = findFiles(RECS_REPO)

MAX_LIMIT = 3000
cnt = 0
totalTime = 0.0

# giter done
for file in file_names:
    trans, diffTime = transcribeThis(RECS_REPO + file)
    cnt = cnt + 1
    totalTime = totalTime + diffTime

    #mycursor = mydb.cursor()

    #sql = "UPDATE recs SET " + col_prefix + " = %s, " + col_prefix + "_ms = %s WHERE filename = %s"
    #val = (trans, diffTime, file)
    #mycursor.execute(sql, val)

    #mydb.commit()

    if cnt >= MAX_LIMIT:
        break


print("Processed",cnt,"recs in", totalTime)
