import functools
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


class EmbeddedCommentsDataset(Dataset):
    '''
    Create ngrams with context and index of predicted words using pretrained Word2Vec dataset
    '''

    def __format_sentences(self, inp: str) -> str:
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

    def __init__(self, base_path, subreddit, context_size=2, remove_stopwords=False):
        super().__init__()

        self.word2vec_model = KeyedVectors.load_word2vec_format(
            r'../models/GoogleNews-vectors-negative300.bin.gz',
            binary=True
        )

        nltk.download('stopwords')

        self.remove_stopwords = remove_stopwords
        self.stop_words = stopwords.words('english')

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

        # convert the sentences to ngrams, and embed them
        self.data = []
        for sent in sentences:
            ngrams = data_processing.convert_sentence_to_ngrams(
                sent, n_param=context_size+1)

            for inp in ngrams:
                context_embedding = data_processing.embed_words(
                    inp[:-1], self.word2vec_model)
                target_label = data_processing.fetch_index(
                    inp[-1], self.word2vec_model)

                if context_embedding is not None and target_label is not None:
                    self.data.append(
                        (context_embedding, torch.LongTensor([target_label]))
                    )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.data[idx]


if __name__ == '__main__':
    obj = CommentsDataset('../dataset/small',
                          'the_donald', remove_stopwords=True)

    print(obj.__len__())

    for i in range(10):
        print(obj.__getitem__(i))
