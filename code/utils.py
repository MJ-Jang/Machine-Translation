import string
import numpy as np
import tensorflow as tf
import os
import pickle

# length percentile of corpus
def length_percentile(tokenized_corpus):
    length = []
    for i in range(len(tokenized_corpus)):
        length.append(len(tokenized_corpus[i]))

    return([min(length),np.percentile(length,25),np.percentile(length,50),
            np.percentile(length,75),np.percentile(length,95),max(length)])

# index of long sentence
def idx_long(corpus, upperbound):
    idx = []
    for i in range(len(corpus)):
        if len(corpus[i]) > upperbound:
            idx.append(i)
    return(idx)

# index of short sentence
def idx_short(corpus, lowerbound):
    idx = []
    for i in range(len(corpus)):
        if len(corpus[i]) <= lowerbound:
            idx.append(i)
    return(idx)

# change to lower alphabat
def sent2lower(corpus):
    for i in range(len(corpus)):
        corpus[i] = corpus[i].lower()

    return(corpus)

# remove punctuations
def remove_punc(corpus):
    translation = str.maketrans("", "", string.punctuation);
    for i in range(len(corpus)):
        corpus[i] = corpus[i].translate(translation)

    return(corpus)

# split by space
def split_space(corpus,remove_last=True,remove_first=True):
    for i in range(len(corpus)):
        corpus[i] = corpus[i].split(" ")

        if remove_first == True:
            # remove first ''
            del(corpus[i][0])

        if remove_last == True:
            # remove last ''
            del(corpus[i][-1])

    return(corpus)

# ADD START, END token
def add_EOS(corpus):
    for i in range(len(corpus)):
        corpus[i] = corpus[i] + ['EOS']

    return(corpus)

# fix the sentence length
def fix_setlength(corpus, limit_length):
    result = corpus
    for i in range(len(corpus)):
        NONE_tokens = ["PAD"]*(limit_length-len(corpus[i]))
        result[i] = result[i] + NONE_tokens

    return (result)

# word2index
def word2index(sentence,word2vec_model):
    # Find index of word in lookup table
    index = []
    for i in range(len(sentence)):
        try:
            index.append(word2vec_model.wv.index2word.index(sentence[i]) + 1)
        except:
            index.append(word2vec_model.wv.index2word.index("UNK") + 1)
    return(index)

# index2word
def index2word(index_seq,word2vec_model):
    # Find index of word in lookup table
    word = []
    for i in range(len(index_seq)):
            word.append(word2vec_model.wv.index2word[index_seq[i]-1])

    return(word)

# change to unk token
def change2UNK(corpus,word2vec_model):
    for i in range(len(corpus)):
        for j in range(len(corpus[i])):
            try:
                tmp = word2vec_model.wv[corpus[i][j]]
            except:
                corpus[i][j] = 'UNK'

        if i % 10000 == 0:
            print("Finished : ",i)


# Onehot encoder
def OnehotEncoder(sentence,word2vec_model):
    # Find index of word in lookup table
    index = []
    for i in range(len(sentence)):
        index.append(word2vec_model.wv.index2word.index(sentence[i]))

    # change index to onehot
    onehot = tf.one_hot(indices=index,depth=len(word2vec_model.wv.index2word))
    return(onehot)

# vector encoder
def wordvec(sentence,word2vec_model):
    vector = []
    for i in range(len(sentence)):
        vector.append(word2vec_model.wv[sentence[i]])

    # change index to onehot
    return (vector)

# batch pad
def batch_pad(X,limit_length,input_dim,reverse=False):

    result = np.array([x[:limit_length] for x in X])
    for i in range(len(X)):
        l = len(result[i])
        zeros = np.tile([0]*input_dim,[limit_length-l,1])
        if reverse:
            result[i] = np.r_[result[i], zeros][::-1]  # zero padding in front
        else:
            result[i] = np.r_[result[i],zeros] # zero padding in front
        result = [x for x in result]

    return result

# load function
def load(path):
    with open(path, "rb") as fp:  # Unpickling
        tmp = pickle.load(fp)
    return tmp

# save function
def save(path,file):
    with open(path, "wb") as fp:
        pickle.dump(file, fp)