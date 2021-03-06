import os
import pandas as pd
import string
import gensim
import collections
from nltk.tokenize import TweetTokenizer

path = os.getcwd()
word2vec = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
index2word_set = word2vec.vocab


#path = "C:/Users/Rajath Kashyap/Downloads/Yardbird Southern Table & Bar_faPVqws-x-5k2CQKDNtHxw.txt"

def word2vec_matrix(word_list):
    words = []
    for word in (word_list):
        if word in index2word_set:
#            words.append({'word':word, 'vector':word2vec[word]})
            words.append({'word':word, 'word2vec':word2vec[word]})
    return pd.DataFrame(words)


def read_reviews(file):
     with open (file, "r", encoding="utf-8") as f:
         return [x.strip() for x in f.readlines()]
    
    
def pre_process(data):
    word_tokenizer = TweetTokenizer()
    words = list()
    table = str.maketrans('', '', string.punctuation)
    for sentence in data:
        for word in word_tokenizer.tokenize(sentence):
            if word.lower() in index2word_set and len(word.strip()) > 0:
                words.append(word.translate(table).lower())                
    word_counter = collections.Counter(words)
    return words, word_counter            
    

files = os.listdir(path = path+'\\reviews')
files = [path+'\\'+f for i in files]

data = read_reviews(files[0])
data, word_count = pre_process(data)
df = word2vec_matrix(word_count)
