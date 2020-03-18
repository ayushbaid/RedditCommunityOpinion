import functools
import os
import operator
import re

import nltk
import numpy as np
import torch
from nltk.corpus import stopwords

from torch.utils.data import Dataset


class CommentsDataset(Dataset):
    def __init__(self, base_path, subreddit, remove_stopwords=False, eos_token=None):
        super().__init__()

        self.eos_token = eos_token

        nltk.download('stopwords')

        self.remove_stopwords = remove_stopwords
        self.stop_words = stopwords.words('english')

        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

        # load the full numpy array
        npy_path = os.path.join(base_path, subreddit+'.npy')
        raw_data = np.load(npy_path)

        # process the taw data into sentences

        self.sentences = functools.reduce(
            operator.iconcat, [tokenizer.tokenize(inp) for inp in raw_data], [])

        self.sentences = list(map(self.__format_sentences,
                                  filter(lambda x: x !=
                                         '[removed]', self.sentences)
                                  ))

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

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.eos_token is None:
            return self.sentences[idx]

        return self.sentences[idx] + ' ' + self.eos_token

    def get_training_set(self):
        return self.sentences


if __name__ == '__main__':
    obj = CommentsDataset('../dataset/small',
                          'the_donald',
                          remove_stopwords=True,
                          eos_token='</s>')

    print(obj.__len__())

    for i in range(10):
        print(obj.__getitem__(i))
