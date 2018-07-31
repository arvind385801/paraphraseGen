import collections
import os
import re

import numpy as np
from six.moves import cPickle

from .functional import *

idx_files = ['data/words_vocab.pkl',
                      'data/characters_vocab.pkl']

[idx_to_word, idx_to_char] = [cPickle.load(open(file, "rb")) for file in idx_files]
[word_to_idx, char_to_idx] = [dict(zip(idx, range(len(idx)))) for idx in
                                        [idx_to_word, idx_to_char]]

max_word_len = np.amax([len(word) for word in idx_to_word])

def encode_characters(characters):
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

    data_words = [[line.split() for line in target.split('\n')] for target in data]
    data_words = [[[word for word in target if word in idx_to_word] for target in yo] for yo in data_words]

    word_tensor = np.array(
        [[list(map(word_to_idx.get, line)) for line in target] for target in data_words])
    np.save(tensor_files[0][0], word_tensor[0])
    # print(word_tensor.shape)
    character_tensor = np.array(
        [[list(map(encode_characters, line)) for line in target] for target in data_words])
    np.save(tensor_files[1][0], character_tensor[0])
