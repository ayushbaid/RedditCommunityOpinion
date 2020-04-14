import torch
import torch.optim as optim

from nltk.tokenize import wordpunct_tokenize
from torch.utils.data import DataLoader

from data.embedded_comments_dataset import EmbeddedCommentsDataset
from models.word2vec import Word2VecModel

from helper.data_processing import embed_words

if __name__ == '__main__':
    is_cuda = True and torch.cuda.is_available()

    context_size = 2

    with open('../config/subreddits.txt') as f:
        subreddits = [x.strip() for x in f.readlines()]

    subreddits = filter(None, subreddits)

    # load the sample phrases for evaluation
    with open('../config/questions_wv.txt') as f:
        # get the phrases
        phrases = [x.strip() for x in f.readlines()]
        phrases = list(filter(None, phrases))

    for subreddit in subreddits:
        print('Subreddit: ', subreddit)

        result_file = open(
            '../results/word2vec/{}_completion.txt'.format(subreddit), 'w')
        result_file.write('input,output\n')

        # load the data and set up training
        dataset_obj = EmbeddedCommentsDataset(
            '../dataset/',
            subreddit,
            remove_stopwords=True,
            context_size=context_size
        )

        vocab_size = dataset_obj.get_vocab_size()
        print('Size of the vocabulary: {}'.format(vocab_size))

        print('Num entries in the dataset: {}'.format(dataset_obj.__len__()))

        dataloader = DataLoader(dataset_obj, batch_size=32,
                                shuffle=True, num_workers=1, pin_memory=is_cuda)

        model = Word2VecModel(vocab_size, context_size=context_size)

        if is_cuda:
            model = model.cuda()

        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5)

        num_iters = int(30)

        loss_history = []

        for epoch in range(num_iters):

            if epoch % 5 == 1:
                print('{}/{} epochs completed processed'.format(epoch, num_iters))

                if len(loss_history):
                    print('Loss: {}'.format(loss_history[-1]))
            #     with torch.no_grad():
            #         try:
            #             context = embed_words(
            #                 ['trump', 'is'], dataset_obj.word2vec_model
            #             )

            #             if is_cuda:
            #                 result_idx = torch.argmax(
            #                     model(context.cuda())).cpu().item()
            #             else:
            #                 result_idx = torch.argmax(model(context)).cpu().item()

            #             result_word = dataset_obj.word2vec_model.wv.index2word[result_idx]

            #             print('Sample eval: "trump is {}"'.format(result_word))
            #         except:
            #             print('Cannot evaluate sample phrase')

            total_loss = 0

            for batch_idx, (X, y) in enumerate(dataloader):

                # if batch_idx % 250 == 0:
                #     print('\t{}  batches processed'.format(batch_idx))

                model.zero_grad()

                if is_cuda:
                    loss = model.loss_ngrams(
                        X.cuda(), torch.squeeze(y, dim=1).cuda())
                else:
                    loss = model.loss_ngrams(
                        X, torch.squeeze(y, dim=1)
                    )

                if loss is None:
                    continue

                loss.backward()
                optimizer.step()

                total_loss += loss.cpu().item()

            scheduler.step()
            loss_history.append(total_loss)

        # format and tokenize the phrases
        for inp_phrase in phrases:
            formatted_phrases = dataset_obj.format_sentences(inp_phrase)

            # tokenize it
            tokenized_phrases = wordpunct_tokenize(formatted_phrases)

            if len(tokenized_phrases) > context_size:
                continue

            if len(tokenized_phrases) < context_size:
                tokenized_phrases = [
                    '<UNK>']*(
                        context_size-len(tokenized_phrases)
                ) + tokenized_phrases

            try:
                context = embed_words(
                    tokenized_phrases, dataset_obj.word2vec_model
                )

                if is_cuda:
                    result_idx = torch.argmax(
                        model(context.cuda())).cpu().item()
                else:
                    result_idx = torch.argmax(model(context)).cpu().item()

                result_word = dataset_obj.word2vec_model.wv.index2word[result_idx]

                result_file.write(
                    inp_phrase + ',' + ' '.join(tokenized_phrases+[result_word]) + '\n')
            except:
                pass

        result_file.close()
