import os
import math
import time

import torch

from data.wlm import WLMCorpus
from models.wlm_lstm import LSTMModel


class WLMRunner():
    '''
    Runner for wlm_lstm model, providing train and test code
    '''

    def __init__(self, subreddit, load_from_disk=True):
        self.emsize = 200  # size of word embeddings
        self.nlayers = 1  # number of layers in the RNN module
        self.lr = 20  # initial learning rate
        self.clip = 0.25  # gradient clipping
        self.num_epochs = 40  # upper epoch limit
        self.batch_size = 20  # batch size
        self.bptt = 5  # bptt sequence length

        self.max_recon = 15

        self.is_cuda = True and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.is_cuda else 'cpu')

        self.model_path = '../models/wlm_lstm/{}.pt'.format(subreddit)

        self.corpus = WLMCorpus('../dataset/100k/', subreddit)

        # load the sample phrases for evaluation
        with open('../config/questions.txt') as f:
            # get the phrases
            phrases = [x.strip() for x in f.readlines()]

            phrases = list(filter(None, phrases))

        # add to the corpus
        self.phrase_ids = list(map(self.corpus.tokenize_normal, phrases))
        # self.phrase_ids = self.corpus.tokenize_normal()

        self.ntokens = len(self.corpus.dictionary)

        self.model = LSTMModel(self.ntokens, self.emsize).to(self.device)

        self.train_data = self.batchify(self.corpus.data_ids, self.batch_size)

        self.log_interval = 5000  # log after # batches

        self.criterion = torch.nn.NLLLoss()

        # load the model from disk if it exists
        if os.path.exists(self.model_path) and load_from_disk:
            checkpoint = torch.load(self.model_path, map_location=self.device)

            # TODO: just storing the model for now
            self.model.load_state_dict(checkpoint['model_state_dict'])

        if not os.path.exists(os.path.dirname(self.model_path)):
            os.makedirs(os.path.dirname(self.model_path))

    def save_model(self):
        torch.save({
            'model_state_dict': self.model.state_dict()
        }, self.model_path)

    def batchify(self, data, bsz):
        # Starting from sequential data, batchify arranges the dataset into columns.
        # For instance, with the alphabet as the sequence and batch size 4, we'd get
        # ┌ a g m s ┐
        # │ b h n t │
        # │ c i o u │
        # │ d j p v │
        # │ e k q w │
        # └ f l r x ┘.
        # These columns are treated as independent by the model, which means that the
        # dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
        # batch processing.

        # Work out how cleanly we can divide the dataset into bsz parts.
        nbatch = data.size(0) // bsz
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, nbatch * bsz)
        # Evenly divide the data across the bsz batches.
        data = data.view(bsz, -1).t().contiguous()
        return data.to(self.device)

    def repackage_hidden(self, h):
        """Wraps hidden states in new Tensors, to detach them from their history."""

        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(self.repackage_hidden(v) for v in h)

    # def evaluate(self, data_source):
    #     # Turn on evaluation mode which disables dropout.
    #     self.model.eval()
    #     total_loss = 0.
    #     ntokens = len(self.corpus.dictionary)

    #     hidden = self.model.init_hidden(eval_batch_size)
    #     with torch.no_grad():
    #         for i in range(0, data_source.size(0) - 1, self.bptt):
    #             data, targets = get_batch(data_source, i)

    #             output, hidden = model(data, hidden)
    #             hidden = repackage_hidden(hidden)
    #             output_flat = output.view(-1, ntokens)
    #             total_loss += len(data) * \
    #                 criterion(output_flat, targets).item()
    #     return total_loss / (len(data_source) - 1)s

    def get_batch(self, source, i):
        # get_batch subdivides the source data into chunks of length args.bptt.
        # If source is equal to the example output of the batchify function, with
        # a bptt-limit of 2, we'd get the following two Variables for i = 0:
        # ┌ a g m s ┐ ┌ b h n t ┐
        # └ b h n t ┘ └ c i o u ┘
        # Note that despite the name of the function, the subdivison of data is not
        # done along the batch dimension (i.e. dimension 1), since that was handled
        # by the batchify function. The chunks are along dimension 0, corresponding
        # to the seq_len dimension in the LSTM.

        seq_len = min(self.bptt, len(source) - 1 - i)
        data = source[i:i+seq_len]
        target = source[i+1:i+1+seq_len].view(-1)
        return data, target

    def train_iter(self, epoch):
        self.model.train()

        total_loss = 0.0

        start_time = time.time()

        hidden = self.model.init_hidden(self.batch_size)

        for batch_idx, i in enumerate(range(0, self.train_data.size(0)-1, self.bptt)):
            # data, targets = self.get_batch(self.train_data, i)
            data, targets = self.get_batch(self.train_data, i)

            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            self.model.zero_grad()

            hidden = self.repackage_hidden(hidden)
            output, hidden = self.model(data, hidden)

            loss = self.criterion(output.view(-1, self.ntokens), targets)

            loss.backward()

            torch.nn.utils.clip_grad_norm(self.model.parameters(), self.clip)

            for p in self.model.parameters():
                p.data.add_(-self.lr, p.grad.data)

            total_loss += loss.item()

            if batch_idx % self.log_interval == 0 and batch_idx > 0:
                cur_loss = total_loss / self.log_interval
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                      'loss {:5.2f} | ppl {:8.2f}'.format(
                          epoch, batch_idx, len(
                              self.train_data) // self.bptt, self.lr,
                          elapsed * 1000 / self.log_interval, cur_loss, math.exp(cur_loss)))
                total_loss = 0
                start_time = time.time()

    def train(self):
        # try:
        for epoch in range(self.num_epochs):
            epoch_start_time = time.time()
            self.train_iter(epoch+1)
            self.save_model()

            if epoch % 5 == 0:
                print('-' * 89)
                self.evaluate_phrases()
                print('| end of epoch {:3d} | time: {:5.2f}s | '.format(
                    epoch+1, (time.time() - epoch_start_time)))
                print('-' * 89)
            # Save the model if the validation loss is the best we've seen so far.
            # if not best_val_loss or val_loss < best_val_loss:
            #     with open(args.save, 'wb') as f:
            #         torch.save(model, f)
            #     best_val_loss = val_loss
            # else:
            #     # Anneal the learning rate if no improvement has been seen in the validation dataset.
            if epoch % 2 == 0 and epoch > 0:
                # Anneal the learning rate
                self.lr /= 4.0
        # except:
        #     pass

    def evaluate_phrases(self):
        self.model.eval()

        reconstructions = []
        with torch.no_grad():
            for inp_ids in self.phrase_ids:

                recon = []

                hidden = self.model.init_hidden(1)
                # feed the exising input ids to the model
                output = None
                for id in inp_ids:
                    output, hidden = self.model.forward(
                        id.to(self.device).view(1, 1), hidden)

                    recon.append(
                        self.corpus.lookup_word(
                            id.unsqueeze(dim=0).detach().cpu().item())
                    )

                # keep on feeding the output till the output is not EOS

                curr_idx = torch.argmax(
                    output.unsqueeze(dim=0).detach().cpu()).item()

                curr_word = self.corpus.lookup_word(curr_idx)

                recon.append(curr_word)

                while len(recon) <= self.max_recon and curr_word != self.corpus.eos_token:
                    id = torch.LongTensor([curr_idx]).to(
                        self.device).view(1, 1)

                    output, hidden = self.model.forward(
                        id, hidden
                    )

                    curr_idx = torch.argmax(
                        output.unsqueeze(dim=0).detach().cpu()).item()

                    curr_word = self.corpus.lookup_word(curr_idx)

                    recon.append(curr_word)

                reconstructions.append(recon)

        print('Phrase evaluations:')
        for words in reconstructions:
            print(' '.join(words))


if __name__ == '__main__':
    # import pdb
    # pdb.set_trace()

    obj = WLMRunner('the_donald', load_from_disk=False)

    obj.train()