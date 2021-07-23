import re
import math
from operator import truediv
from collections import Counter
from itertools import combinations
import numpy as np
import pandas as pd
import nltk
nltk.download("popular")
nltk.download('universal_tagset')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import heapq
from sklearn.feature_extraction.text import CountVectorizer
from textblob import TextBlob
import collections
import textstat
from sklearn.preprocessing import normalize

meta = pd.read_csv("metadata.txt",  sep='delimiter', engine='python')
m = []
for i in range(len(meta)):
  m.append(meta.loc[i,:].values[0].split())
fin = pd.DataFrame(columns=["Date", "review", "reviewer", "productID", "Label", "Useful", "Funny", "Cool", "star"], data=m)
# print(fin)

review = pd.read_csv("review.txt", sep='delimiter',  engine='python')
r = review.values.flatten()
# print(len(r))
fin.review = r
c = fin.copy()
del c["Date"]
del c["reviewer"]
del c["productID"]
del c["Useful"]
del c["Funny"]
del c["Cool"]
del c["star"]
# print(c.info())

def word_count(s):
  word_tokens = word_tokenize(s)
  x = [i for i in word_tokens if i.isalnum()]
  y = []
  for i in x:
    c = ""
    for j in i:
      if j.isalnum():
        c = c+j
    y.append(c)
  return y

def capital_letters(s):
  c = 0
  x = word_count(s)
  for i in x:
    if i[0].isupper():
      c = c + 1
  return c/len([i for i in s if i.isalnum()])

def capital_words(s):
  c = 0
  x = word_count(s)
  for i in x:
    if i.isupper():
      c = c + 1
  return c/len(x)

def first_person(s):
  c = 0
  f = ["we", "ourselves", "our", "ours", "us", "I", "me", "my", "mine", "myself"]
  x = word_count(s)
  for i in x:
    if i in f:
      c = c+1
  return c/len(x)

def excl_sent(s):
  c = 0
  x = sent_tokenize(s)
  for i in x:
    if "!" in i:
      c = c+1
  return c/len(x)


def bag_of_words(s):
  dataset = nltk.sent_tokenize(s)
  for i in range(len(dataset)):
    dataset[i] = dataset[i].lower()
    dataset[i] = re.sub(r'\W', ' ', dataset[i])
    dataset[i] = re.sub(r'\s+', ' ', dataset[i])

  word2count = {}
  for data in dataset:
    words = nltk.word_tokenize(data)
    for word in words:
      if word not in word2count.keys():
        word2count[word] = 1
      else:
        word2count[word] += 1

  freq_words = word2count

  X = []
  for data in dataset:
    vector = []
    for word in freq_words:
      if word in nltk.word_tokenize(data):
        vector.append(1)
      else:
        vector.append(0)
    X.append(vector)
  X = np.asarray(X)
  return X

def tf_idf(s):
  corpus = nltk.sent_tokenize(s)
  vectorizer = TfidfVectorizer(stop_words='english')
  X = vectorizer.fit_transform(corpus)
  return X

def word_vector(s):
  vectorizer = CountVectorizer()
  vector = vectorizer.fit_transform(s)
  return vector

def pos_tags(s):
  tag = {"ADJ": 0, "ADP": 0, "ADV": 0, "CONJ": 0, "DET": 0, "NOUN": 0, "NUM": 0, "PRT": 0, "PRON": 0, "VERB": 0, ".": 0, "X": 0}
  lower_case = s.lower()
  tokens = nltk.word_tokenize(lower_case)
  tags = nltk.pos_tag(tokens, tagset="universal")
  counts = dict(Counter(tag for word, tag in tags))
  for i,j in dict(collections.OrderedDict(sorted(counts.items()))).items():
    tag[i] = j
  return list(tag.values())

def characteristic(s):
  return [list(TextBlob(s).sentiment)[0], list(TextBlob(s).sentiment)[1], textstat.flesch_kincaid_grade(s), textstat.flesch_reading_ease(s), textstat.gunning_fog(s), textstat.smog_index(s), textstat.automated_readability_index(s), textstat.coleman_liau_index(s), textstat.linsear_write_formula(s), textstat.dale_chall_readability_score(s)]

X_train, X_test, y_train, y_test = train_test_split(c["review"], c["Label"], shuffle=False,stratify = None, test_size=0.25)
vectorizer = CountVectorizer()
X_traincv = vectorizer.fit_transform(X_train)
X_testcv = vectorizer.transform(X_test)

# print(X_train.shape, X_test.shape)
# print(X_traincv.shape, X_testcv.shape)

train = []
test = []


for i, z in zip(X_train, X_traincv):
  ch = list(z.toarray().flatten())
  ch.append(len(word_count(i)))
  ch.append(capital_letters(i))
  ch.append(capital_words(i))
  ch.append(first_person(i))
  ch.append(excl_sent(i))
  for j in pos_tags(i):
    ch.append(j)
  for k in characteristic(i):
    ch.append(k)
  ch = list(np.array(ch)/np.linalg.norm(np.array(ch)))
  train.append(ch)
  print(np.array(train).shape)

print("\n", "############################################")
train = np.array(train)
print(train.shape)
y_train = np.array(y_train).flatten()
print(y_train.shape)
print("############################################")

for i, z in zip(X_test, X_testcv):
  ch = list(z.toarray().flatten())
  ch.append(len(word_count(i)))
  ch.append(capital_letters(i))
  ch.append(capital_words(i))
  ch.append(first_person(i))
  ch.append(excl_sent(i))
  for j in pos_tags(i):
    ch.append(j)
  for k in characteristic(i):
    ch.append(k)
  ch = list(np.array(ch)/np.linalg.norm(np.array(ch)))
  test.append(ch)
  print(np.array(test).shape)

print("\n","############################################")
test = np.array(test)
print(test.shape)
y_test = np.array(y_test).flatten()
print(y_test.shape)
print("############################################")


model = XGBClassifier()
model.fit(train, y_train)
y_pred=model.predict(test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
