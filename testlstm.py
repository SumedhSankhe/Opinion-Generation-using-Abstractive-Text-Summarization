# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 22:24:41 2019

@author: sumed
"""

from __future__ import print_function
#import Keras library
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM, Input, Bidirectional
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.metrics import categorical_accuracy

#import spacy, and spacy french model
# spacy is used to work on text
import spacy
nlp = spacy.load("en_core_web_sm")
#nlp.max_length(x='127810')
#import other libraries
import numpy as np
import random
import sys
import os
import time
import codecs
import collections
from six.moves import cPickle
import glob


data_dir = 'D:/Northeastern_University/CS6120/Project/preped/*'# data directory containing raw texts
save_dir = 'save/' # directory to store trained NN models
file_list = glob.glob(data_dir)
vocab_file = os.path.join(save_dir, "words_vocab.pkl")
sequences_step = 1 #step to create sequences


def create_wordlist(doc):
    wl = []
    for word in doc:
        if word.text not in ("\n","\n\n",'\u2009','\xa0'):
            wl.append(word.text.lower())
    return wl

wordlist = []

for file_name in file_list[:20]:
    input_file = file_name
    #read data
    with codecs.open(input_file, "r", encoding = 'utf8') as f:
        data = f.read()
        
    #create sentences
    doc = nlp(data)
    wl = create_wordlist(doc)
    wordlist = wordlist + wl


# count the number of words
word_counts = collections.Counter(wordlist)

# Mapping from index to word : that's the vocabulary
vocabulary_inv = [x[0] for x in word_counts.most_common()]
vocabulary_inv = list(sorted(vocabulary_inv))

# Mapping from word to index
vocab = {x: i for i, x in enumerate(vocabulary_inv)}
words = [x[0] for x in word_counts.most_common()]

#size of the vocabulary
vocab_size = len(words)
print("vocab size: ", vocab_size)

#save the words and vocabulary
with open(os.path.join(vocab_file), 'wb') as f:
    cPickle.dump((words, vocab, vocabulary_inv), f)
    
    
#create sequences
sequences = []
next_words = []
seq_length =10
for i in range(0, len(wordlist) - seq_length, sequences_step):
    sequences.append(wordlist[i: i + seq_length])
    next_words.append(wordlist[i + seq_length])

print('nb sequences:', len(sequences))


X = np.zeros((len(sequences), seq_length, vocab_size), dtype=np.bool)
y = np.zeros((len(sequences), vocab_size), dtype=np.bool)
for i, sentence in enumerate(sequences):
    for t, word in enumerate(sentence):
        X[i, t, vocab[word]] = 1
    y[i, vocab[next_words[i]]] = 1


def bidirectional_lstm_model(seq_length, vocab_size):
    print('Build LSTM model.')
    model = Sequential()
    model.add(Bidirectional(LSTM(rnn_size, activation="relu"),input_shape=(seq_length, vocab_size)))
    model.add(Dropout(0.6))
    model.add(Dense(vocab_size))
    model.add(Activation('softmax'))
    
    optimizer = Adam(lr=learning_rate)
    callbacks=[EarlyStopping(patience=2, monitor='val_loss')]
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=[categorical_accuracy])
    print("model built!")
    return model

rnn_size = 256 # size of RNN
seq_length =10 # sequence length
learning_rate = 0.001 #learning rate

md = bidirectional_lstm_model(seq_length, vocab_size)
md.summary()

batch_size = 32 # minibatch size
num_epochs = 5 # number of epochs

callbacks=[EarlyStopping(patience=4, monitor='val_loss'),
           ModelCheckpoint(filepath=save_dir + "/" + 'my_model_gen_sentences.{epoch:02d}-{val_loss:.2f}.hdf5',\
                           monitor='val_loss', verbose=0, mode='auto', period=2)]
#fit the model
history = md.fit(X, y,
                 batch_size=batch_size,
                 shuffle=True,
                 epochs=num_epochs,
                 callbacks=callbacks,
                 validation_split=0.1)

#save the model
md.save(save_dir + "/" + 'my_model_generate_sentences.h5')


#load vocabulary
print("loading vocabulary...")
vocab_file = os.path.join(save_dir, "words_vocab.pkl")

with open(os.path.join(save_dir, 'words_vocab.pkl'), 'rb') as f:
        words, vocab, vocabulary_inv = cPickle.load(f)

vocab_size = len(words)

from keras.models import load_model
# load the model
print("loading model...")
model = load_model(save_dir + "/" + 'my_model_generate_sentences.h5')

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

generated =''

words_number = 20 # number of words to generate
seed_sentences = "Bad"
#we shate the seed accordingly to the neural netwrok needs:
for i in range (seq_length):
    sentence.append("a")

seed = seed_sentences.split()

for i in range(len(seed)):
    sentence[seq_length-i-1]=seed[len(seed)-i-1]

generated += ' '.join(sentence)

#the, we generate the text
for i in range(words_number):
    #create the vector
    x = np.zeros((1, seq_length, vocab_size))
    for t, word in enumerate(sentence):
        x[0, t, vocab[word]] = 1.

    #calculate next word
    preds = model.predict(x, verbose=0)[0]
    next_index = sample(preds, 0.33)
    next_word = vocabulary_inv[next_index]

    #add the next word to the text
    generated += " " + next_word
    # shift the sentence by one, and and the next word at its end
    sentence = sentence[1:] + [next_word]
    print(sentence)

#print the whole text
print(generated)

import h5py
filename = 'save/my_model_gen_sentences.02-5.10.hdf5'
f = h5py.File(filename, 'r')
