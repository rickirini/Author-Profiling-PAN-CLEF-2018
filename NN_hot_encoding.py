import numpy as np
np.random.seed(113) #set seed before any keras import
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.utils import np_utils
import sys
import random
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from collections import defaultdict
from keras.callbacks import EarlyStopping
from string import punctuation
from nltk.corpus import stopwords
from keras import optimizers
from keras.optimizers import Adadelta
import string
from keras.constraints import maxnorm
from keras.models import model_from_json
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
import re
import pickle
from keras.models import model_from_yaml
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import StratifiedKFold

def find_files(directory):

    paths = []
    for root, directories, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            paths.append(filepath)
    return [path for path in paths if path.endswith(".out")]

def clean_doc(doc):
    # replace '--' with a space ' '
    doc = doc.replace('--', ' ')
    # split into tokens by white space
    tokens = doc.split()
    # remove punctuation from each token
    table = str.maketrans('', '', string.punctuation)
    tokens = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # make lower case
    tokens = [word.lower() for word in tokens]

    if "@" in tokens:
        tokens ="username"
    if "http" in tokens:
        tokens = "link"
    if "#" in tokens:
        tokens = "hastag"
    accounts = " ".join(tokens)
    return accounts


# first map all words to indices, then create n-hot vector
def convert_to_n_hot(X, vocab_size):
    out = []
    for instance in X:
        n_hot = np.zeros(vocab_size)
        for w_idx in instance:
            n_hot[w_idx] = 1
        out.append(n_hot)
    return np.array(out)



"""
loads the data
"""
en_user_list=[]
ar_user_list=[]
pt_user_list=[]
es_user_list=[]
with open("data/pan18-author-profiling-training-2018-02-27/en/truth.txt", "r") as outfile:
    for line in outfile:
        user = line.rstrip().split(":::")
        en_user_list.append(user)
outfile.close()

with open("data/pan18-author-profiling-training-2018-02-27/ar/truth.txt", "r") as outfile:
    for line in outfile:
        user = line.rstrip().split(":::")
        ar_user_list.append(user)
outfile.close()
with open("data/pan18-author-profiling-training-2018-02-27/es/truth.txt", "r") as outfile:
    for line in outfile:
        user = line.rstrip().split(":::")
        es_user_list.append(user)
outfile.close()

female_users = []
male_users = []
for fl in find_files("data/tweets/" + sys.argv[1]):
    text = open(fl).readlines()
    filename = fl[15:]
    file_name = filename.split('.')

    if sys.argv[1] == "ar":
        userlist = ar_user_list
    elif sys.argv[1] == "en":
        userlist = en_user_list
    elif sys.argv[1] == "es":
        userlist = es_user_list

    for i in userlist:
        if i[0] == file_name[0]:
            text.append(i)
    if text[-1][1] == "male":
        all_tweet = " ".join(text[:-1])
        all_tweets= clean_doc(all_tweet)
        male_users.append(all_tweets)
    else:
        all_tweet = " ".join(text[:-1])
        all_tweets= clean_doc(all_tweet)
        female_users.append(all_tweets)

le = preprocessing.LabelEncoder()

male_account_labeld = \
[(profile, "male" )
    for profile in male_users]    
female_account_labeld = \
[(profile, "female")
    for profile in female_users] 

#seperate the labels and profiles
male_labels = [labels for accounts,labels in male_account_labeld]
female_labels = [labels for accounts,labels in female_account_labeld]


male_lines = [accounts for accounts,labels in male_account_labeld]
female_lines = [accounts for accounts,labels in female_account_labeld]

#concatenate all data
sentences = np.concatenate([male_lines, female_lines ], axis=0)
labels = np.concatenate([male_labels, female_labels ], axis=0)
transform = le.fit(labels)
labels = le.transform(labels)
## make sure we have a label for every data instance
assert(len(sentences)==len(labels))
data={}
np.random.seed(113) #seed
data['target']= np.random.permutation(labels)
np.random.seed(113) # use same seed!
data['data'] = np.random.permutation(sentences)
#transform labels to numeric data


X_train, X_test, y_train, y_test= train_test_split(data['data'], data['target'], test_size=0.2)
# X_train,  y_train = train_test_split(X_rest, y_rest, test_size=0.2)
# del X_rest, y_rest
print("#train instances: {}  #test: {}".format(len(X_train),len(X_test)))


# NEURAL NETWORK WITH Traditional sparse n-hot encoding
w2i = defaultdict(lambda: len(w2i))
UNK = w2i["<unk>"]
# convert words to indices, taking care of UNKs
X_train_num = [[w2i[word] for word in sentence.split(" ")] for sentence in X_train]


w2i = defaultdict(lambda: UNK, w2i) # freeze

X_test_num = [[w2i[word] for word in sentence.split(" ")] for sentence in X_test]

vocab_size = len(w2i)
X_train_nhot = convert_to_n_hot(X_train_num, vocab_size)
X_test_nhot = convert_to_n_hot(X_test_num, vocab_size)



# np.random.seed(113) #set seed before any keras import
# model = Sequential()
# model.add(Dense(100, input_shape=(vocab_size,), kernel_initializer='he_uniform',  kernel_constraint=maxnorm(5)))
# model.add(Activation('relu'))
# model.add(Dropout(0.4))
# model.add(Dense(1))
# model.add(Activation('sigmoid'))
# optimizer =Adadelta(lr=0.22)
# model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
# # plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
# print(model.summary())

# model.fit(X_train_nhot, y_train, epochs=15, verbose=1, batch_size=64)
# loss, accuracy = model.evaluate(X_test_nhot,y_test)

# print("Accuracy: ", accuracy *100)

# 10 fold cross validation
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=113)
cvscores = []
for train, test in kfold.split(X_train_nhot, y_train):
    #create model
    model = Sequential()
    model.add(Dense(100, input_shape=(vocab_size,), kernel_initializer='he_uniform',  kernel_constraint=maxnorm(5)))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    optimizer =Adadelta(lr=0.22)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    # Fit the model
    model.fit(X_train_nhot, y_train, epochs=15, batch_size=64, verbose=0)
    # evaluate the model
    scores = model.evaluate(X_test_nhot,y_test, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

