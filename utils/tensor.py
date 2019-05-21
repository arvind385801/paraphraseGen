import collections
import os
import re

import numpy as np
from six.moves import cPickle

from .functional import *

'''
idx_files = ['data/quora/words_vocab.pkl',
                      'data/quora/characters_vocab.pkl']

[idx_to_word, idx_to_char] = [cPickle.load(open(file, "rb")) for file in idx_files]
[word_to_idx, char_to_idx] = [dict(zip(idx, range(len(idx)))) for idx in
                                        [idx_to_word, idx_to_char]]
'''


def encode_characters(characters,char_to_idx, idx_to_word):

    max_word_len = np.amax([len(word) for word in idx_to_word])
    word_len = len(characters)
    to_add = max_word_len - word_len
    characters_idx = [char_to_idx[i] for i in characters] + to_add * [char_to_idx['']]
    return characters_idx

def preprocess_data(data_files, idx_files, tensor_files, file, str=''):

    # print 'Preprocessing the test file\n'
    if file:
        data = [open(file, "r").read() for file in data_files]
    else:
        data=[str+'\n']

    #gisse added
    [idx_to_word, idx_to_char] = [cPickle.load(open(file, "rb")) for file in idx_files]
    [word_to_idx, char_to_idx] = [dict(zip(idx, range(len(idx)))) for idx in
                                        [idx_to_word, idx_to_char]]


    data_words = [[line.split() for line in target.split('\n')] for target in data]
    data_words = [[[word for word in target if word in idx_to_word] for target in yo] for yo in data_words]

    word_tensor = np.array(
        [[list(map(word_to_idx.get, line)) for line in target] for target in data_words])
    np.save(tensor_files[0][0], word_tensor[0])
    # print(word_tensor.shape)
    character_tensor = np.array(
        [[list(map(lambda p: encode_characters(p, char_to_idx, idx_to_word), line)) for line in target] for target in data_words])
        #[[list(map(encode_characters, line)) for line in target] for target in data_words])
    np.save(tensor_files[1][0], character_tensor[0])
