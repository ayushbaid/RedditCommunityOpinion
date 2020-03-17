import torch
import torch.optim as optim

from torch.utils.data import DataLoader

from data.embedded_comments_dataset import EmbeddedCommentsDataset
from models.word2vec import Word2VecModel

if __name__ == '__main__':
    # load the data and set up training
    dataset_obj = EmbeddedCommentsDataset(
        '../dataset/small/', 'the_donald', remove_stopwords=False)

    print('Num entries in the dataset: {}'.format(dataset_obj.__len__()))

    dataloader = DataLoader(dataset_obj, batch_size=16,
                            shuffle=True, num_workers=1)

    model = Word2VecModel()

    optimizer = optim.SGD(model.parameters(), lr=0.001)

    num_iters = int(1e2)

    loss_history = []

    for epoch in range(num_iters):
        print('{}/{} epochs completed processed'.format(epoch, num_iters))

        total_loss = 0

        for batch_idx, (X, y) in enumerate(dataloader):

            if batch_idx % 10 == 0:
                print('\t{}  batches processed'.format(batch_idx))

            model.zero_grad()
            loss = model.loss_ngrams(X, torch.squeeze(y, dim=1))

            if loss is None:
                continue

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        loss_history.append(total_loss)
