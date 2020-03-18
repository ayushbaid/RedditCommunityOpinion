'''
Code from the pytorch WLM model:
https://github.com/pytorch/examples/tree/master/word_language_model

'''

import functools
import operator
import os
import re

from typing import List


import nltk
import numpy as np
import torch

from nltk.util import ngrams


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class WLMCorpus:
    def __init__(self, base_path: str, subreddit: str, max_sentence_length: int, eos_token='<eos>'):
        super().__init__()

        self.dictionary = Dictionary()
        self.eos_token = eos_token

        self.max_sentence_length = max_sentence_length

        self.dictionary.add_word(self.eos_token)

        self.eos_id = self.dictionary.word2idx[self.eos_token]

        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

        # load the full numpy array
        npy_path = os.path.join(base_path, subreddit+'.npy')
        raw_data = np.load(npy_path)

        # process the taw data into sentences

        sentences = functools.reduce(
            operator.iconcat, [tokenizer.tokenize(inp) for inp in raw_data], [])

        sentences = list(map(self.__format_sentences,
                             filter(lambda x: x !=
                                    '[removed]', sentences)
                             ))

        self.data_ids = self.tokenize_as_ngrams(sentences)

    def __format_sentences(self, inp: str) -> str:
        # format the sentences to remove special characters

        # removing subreddit and user links
        clean_review = re.sub(r'/?(r|u)/\w+/?', '', inp)

        clean_review = re.sub('[^a-zA-Z]', ' ', clean_review)

        # convert to lower case
        clean_review = clean_review.lower()

        return clean_review

    def tokenize_as_ngrams(self, sentences, add_eos_token=True):

        # Add words to the dictionary
        for line in sentences:
            words = line.split()
            for word in words:
                self.dictionary.add_word(word)

        # Tokenize the sentences
        idss = []
        for line in sentences:

            words = line.split()

            if add_eos_token:
                words = words + [self.eos_token]

            # # padd the sentence so that it is divisible by the input_length
            # num_fill = len(words) % self.input_length

            # words = words + [self.eos_token] * \
            #     (self.input_length-num_fill)

            ids = []
            for word in words:
                ids.append(self.dictionary.word2idx[word])

            # reformat the IDs into ngrams
            # id_ngrams = ngrams(
            #     ids, self.input_length+1, pad_right=False)

            # import pdb
            # pdb.set_trace()

            # reformat this into length of
            idss.append(torch.tensor(list(ids)).type(torch.int64))

        ids = torch.cat(idss)

        return ids

    def tokenize_normal(self, sentence: List[str]) -> torch.IntTensor:
        words = sentence.split()

        ids = []
        for word in words:
            ids.append(self.dictionary.add_word(word))

        return torch.tensor(ids).type(torch.int64)

    def lookup_word(self, id: int) -> str:
        return self.dictionary.idx2word[id]
