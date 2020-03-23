'''
Code from the pytorch WLM model:
https://github.com/pytorch/examples/tree/master/word_language_model

'''

import functools
import operator
import os
import pickle
import glob
import re

from typing import List


import nltk
import numpy as np
import torch

from gensim.models import KeyedVectors


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

    def load_w2v_embeddings(self, word2vec_path):
        word2vec_model = KeyedVectors.load_word2vec_format(
            word2vec_path,
            binary=True
        )

        num_words = len(self.idx2word)

        embeddings = np.random.rand(num_words, 300) * 0.2 - 0.1

        for idx, word in enumerate(self.idx2word):
            try:
                embeddings[idx] = word2vec_model[word]
            except KeyError:
                pass

        return torch.from_numpy(embeddings).float()


class WLMCorpus:
    def __init__(self, base_path: str, subreddit: str, max_sentence_length: int, eos_token='<eos>', dictionary_init: dict = None):
        super().__init__()

        self.dictionary = Dictionary()

        if dictionary_init is not None:
            self.dictionary.word2idx = dictionary_init['word2idx']
            self.dictionary.idx2word = dictionary_init['idx2word']

        self.eos_token = eos_token

        self.max_sentence_length = max_sentence_length

        self.dictionary.add_word(self.eos_token)

        self.eos_id = self.dictionary.word2idx[self.eos_token]

        self.tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

        train_file_template = os.path.join(
            base_path, '{}_train_part*.npy'.format(subreddit))
        val_file_template = os.path.join(
            base_path, '{}_val_part*.npy'.format(subreddit))

        print('Train path: ', train_file_template)
        print('Val path: ', val_file_template)

        train_files = glob.glob(train_file_template)
        val_files = glob.glob(val_file_template)

        train_sentences = []
        for file_name in train_files:
            train_sentences += self.__load_file(file_name)

        val_sentences = []
        for file_name in val_files:
            val_sentences += self.__load_file(file_name)

        self.train_data_ids = self.tokenize(train_sentences)
        self.val_data_ids = self.tokenize(val_sentences)

        print("-"*89)
        print("Total training sentences = ", len(train_sentences))
        print("Total validation sentences = ", len(val_sentences))
        print("-"*89)

    def __load_file(self, npy_path):
        raw_data = np.load(npy_path)

        sentences = functools.reduce(
            operator.iconcat, [self.tokenizer.tokenize(inp) for inp in raw_data], [])

        sentences = list(map(self.__format_sentences,
                             filter(lambda x: x !=
                                    '[removed]', sentences)
                             ))

        return sentences

    def __format_sentences(self, inp: str) -> str:
        # format the sentences to remove special characters

        # removing subreddit and user links
        clean_review = re.sub(r'/?(r|u)/\w+/?', '', inp)

        clean_review = re.sub('[^a-zA-Z]', ' ', clean_review)

        # convert to lower case
        clean_review = clean_review.lower()

        return clean_review

    def tokenize(self, sentences, add_eos_token=True):

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

    def save_dictionary(self, save_path):
        print("Saving dictionary to ", save_path)
        with open(save_path, 'wb') as f:
            pickle.dump({
                'word2idx': self.dictionary.word2idx,
                'idx2word': self.dictionary.idx2word
            }, f)
