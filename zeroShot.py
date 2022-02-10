# it works - but not that useful imo
from transformers import pipeline

categories = ['online banking', 'operator', 'check balance']
sequence_to_classify = "I want to talk to representative"

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
result = classifier(sequence_to_classify, categories)

print(result)
