import torch
import torch.optim as optim

from torch.utils.data import DataLoader

from data.embedded_comments_dataset import EmbeddedCommentsDataset
from models.word2vec import Word2VecModel

from helper.data_processing import embed_words

if __name__ == '__main__':
    is_cuda = True and torch.cuda.is_available()

    # load the data and set up training
    dataset_obj = EmbeddedCommentsDataset(
        '../dataset/small/', 'the_donald', remove_stopwords=False)

    vocab_size = dataset_obj.get_vocab_size()
    print('Size of the vocabulary: {}'.format(vocab_size))

    print('Num entries in the dataset: {}'.format(dataset_obj.__len__()))

    dataloader = DataLoader(dataset_obj, batch_size=16,
                            shuffle=True, num_workers=1)

    model = Word2VecModel(vocab_size)

    if is_cuda:
        model = model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=0.001)

    num_iters = int(1e2)

    loss_history = []

    for epoch in range(num_iters):
        print('{}/{} epochs completed processed'.format(epoch, num_iters))

        if epoch % 5 == 0:
            # test on dummy phrase
            if len(loss_history):
                print('Loss: {}'.format(loss_history[-1]))
            with torch.no_grad():
                try:
                    context = embed_words(
                        ['trump', 'is'], dataset_obj.word2vec_model
                    )

                    if is_cuda:
                        result_idx = torch.argmax(
                            model(context.cuda())).cpu().item()
                    else:
                        result_idx = torch.argmax(model(context)).cpu().item()

                    result_word = dataset_obj.word2vec_model.wv.index2word[result_idx]

                    print('Sample eval: "trump is {}"'.format(result_word))
                except:
                    print('Cannot evaluate sample phrase')

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

        loss_history.append(total_loss)
