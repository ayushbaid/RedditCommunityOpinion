from typing import List

import numpy as np
import torch
import torch.nn as nn

from models.base_model import BaseModel


class Word2VecLSTMModel(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()

        embedding_dim = 300

        self.encoding_layer = nn.Linear(embedding_dim, 128)
        self.recurrent_layer = nn.LSTM(128, 128)

        self.decoding_layer = nn.Linear(128, vocab_size)

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, inputs) -> torch.tensor:
        return self.model(inputs)

    # def complete_ngram(self, context: List[str]) -> str:
    #     if len(context) != self.context_size:
    #         raise Exception('Context is of insufficient size')

    #     embedded_context = self.__embed_words(context)
    #     model_output = self.model.forward(embedded_context)

    #     # get the word corresponsind to the maximum probability
    #     predicted_word = self.word2vec_model.wv.index2word[torch.argmax(
    #         model_output)]

    #     return predicted_word

    def loss_ngrams(self, x, index_gt):

        model_output = self.forward(x)

        return self.loss_fn(model_output, index_gt)


# if __name__ == '__main__':
#     obj = Word2VecModel()
#     # print(obj.complete_ngram(['this', 'is']))
