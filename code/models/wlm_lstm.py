'''
Code from the WLM LSTM model
Link: https://github.com/pytorch/examples/blob/master/word_language_model

'''

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ntoken, ninput, nlayers=1):  # dropout=0.5
        super().__init__()

        self.ntoken = ntoken
        self.ninput = ninput
        self.nlayers = nlayers

        #self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninput)

        self.rnn = nn.LSTM(ninput, ninput, nlayers)  # dropout=dropout

        self.decoder = nn.Linear(ninput, ntoken)

        self.init_weights()

        # tying the encoder and decoder weights
        self.decoder.weight = self.encoder.weight

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, inp, hidden):
        emb = self.encoder(inp)

        output, hidden = self.rnn(emb, hidden)

        decoded_output = self.decoder(output).view(-1, self.ntoken)

        return F.log_softmax(decoded_output, dim=1), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())

        return (
            weight.new_zeros(self.nlayers, bsz, self.ninput),
            weight.new_zeros(self.nlayers, bsz, self.ninput),
        )
