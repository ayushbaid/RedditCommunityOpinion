import functools
import glob
import os
import operator
import re

import nltk
import numpy as np
import torch

from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from torch.utils.data import Dataset

import helper.data_processing as data_processing
import helper.w2v as w2v


class EmbeddedCommentsDataset(Dataset):
    '''
    Create ngrams with context and index of predicted words using pretrained Word2Vec dataset
    '''

    def format_sentences(self, inp: str) -> str:
        # format the sentences to remove special characters

        # removing subreddit and user links
        clean_review = re.sub(r'/?(r|u)/\w+/?', '', inp)

        clean_review = re.sub('[^a-zA-Z]', ' ', clean_review)

        # convert to lower case
        clean_review = clean_review.lower()

        if self.remove_stopwords:
            return ' '.join([word for word in clean_review.split() if word not in self.stop_words])
        else:
            return clean_review

    def __init__(self, base_path, subreddit, context_size=2, remove_stopwords=True):
        super().__init__()

        self.word2vec_model = KeyedVectors.load_word2vec_format(
            r'../models/GoogleNews-vectors-negative300.bin.gz',
            binary=True
        )

        # nltk.download('stopwords')

        self.remove_stopwords = remove_stopwords
        self.stop_words = set(stopwords.words('english'))

        # load stop words from disk
        with open('../config/stopwords.txt', 'r') as f:
            manual_stop_words = [x.strip() for x in f.readlines()]

            manual_stop_words = set(filter(None, manual_stop_words))

            self.stop_words.update(manual_stop_words)

        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

        # load the full numpy array
        train_file_template = os.path.join(
            base_path, '{}_train_part*.npy'.format(subreddit))

        data_list = []
        for file_path in glob.glob(train_file_template):
            data_list.append(np.load(file_path))
        raw_data = np.concatenate(data_list, axis=0)
        data_list = None

        # process the raw data into sentences

        sentences = functools.reduce(
            operator.iconcat, [tokenizer.tokenize(inp) for inp in raw_data], [])

        sentences = list(map(self.format_sentences,
                             filter(lambda x: x !=
                                    '[removed]', sentences)
                             ))

        # get the vocab from sentences
        dataset_vocab = data_processing.get_vocab_set(sentences)

        # Add the unknown token to the dataset vocab
        dataset_vocab.add('<UNK>')

        w2v.restrict_w2v(self.word2vec_model, dataset_vocab)

        # convert the sentences to ngrams, and embed them
        self.data = []
        for sent in sentences:
            ngrams = data_processing.convert_sentence_to_ngrams(
                sent, n_param=context_size+1, add_unknown=True)

            for inp in ngrams:
                context_embedding = data_processing.embed_words(
                    inp[:-1], self.word2vec_model)
                target_label = data_processing.fetch_index(
                    inp[-1], self.word2vec_model)

                if context_embedding is not None and target_label is not None:
                    self.data.append(
                        (context_embedding, torch.LongTensor([target_label]))
                    )

    def get_vocab_size(self):
        return len(self.word2vec_model.wv.vocab)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.data[idx]


if __name__ == '__main__':
    obj = EmbeddedCommentsDataset('../dataset/small',
                                  'the_donald', remove_stopwords=True)

    print(obj.__len__())

    for i in range(10):
        print(obj.__getitem__(i))
