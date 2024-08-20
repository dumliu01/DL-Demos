import string
import time

import numpy as np
import torch
import torch.nn as nn

from dldemos.Transformer_ver1.dataset import tensor_to_sentence
from dldemos.Transformer_ver1.model import Transformer
from dldemos.Transformer_ver1.preprocess_data import (PAD_ID, get_dataloader,
                                                 load_sentences, load_vocab)

# Config
batch_size = 64
lr = 0.0001
d_model = 512
d_ff = 2048
n_layers = 6
heads = 8
dropout_rate = 0.2
n_epochs = 60
PAD_ID = 0
maxlen = 200

def main():
    en_vocab, zh_vocab = load_vocab()

    en_train, zh_train, en_valid, zh_valid = load_sentences()
    dataloader_train = get_dataloader(en_train, zh_train, batch_size)

    print_interval = 1000
    device_id = 0



    model = Transformer(len(en_vocab), len(zh_vocab),PAD_ID, d_model, d_ff,
                        n_layers, heads, dropout_rate, maxlen)
    #model.to(device_id)
    device = torch.device("cuda:0")  # 指定使用第一个GPU
    model.to(device)

    #model.init_weights()

    optimizer = torch.optim.Adam(model.parameters(), lr)
    citerion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    tic = time.time()
    cnter = 0
    dataset_len = len(dataloader_train.dataset)
    print('Dataset size:', dataset_len)
    for epoch in range(10):
        loss_sum = 0

        for x, y in dataloader_train:
            x, y = x.to(device_id), y.to(device_id)
            x_mask = x == PAD_ID
            y_mask = y == PAD_ID
            y_input = y[:, :-1]
            y_label = y[:, 1:]
            y_mask = y_mask[:, :-1]
            #y_hat = model(x, y_input, x_mask, y_mask)
            y_hat = model(x, y_input)

            n, seq_len = y_label.shape
            y_hat = torch.reshape(y_hat, (n * seq_len, -1))
            y_label = torch.reshape(y_label, (n * seq_len, ))
            y_label = y_label.to(torch.int64)
            loss = citerion(y_hat, y_label)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            loss_sum += loss.item()

            toc = time.time()
            interval = toc - tic
            minutes = int(interval // 60)
            seconds = int(interval % 60)
            if cnter % print_interval == 0:
                print(f'{cnter:08d} {minutes:02d}:{seconds:02d}'
                      f' loss: {loss.item()}')
            cnter += 1

        print(f'Epoch {epoch}. loss: {loss_sum / dataset_len}')
        savepath = "saved/model-{0}.pt".format(epoch)
        torch.save(model.state_dict(), savepath)

    torch.save(model.state_dict(), 'model.pth')
    print('Done.')

if __name__ == '__main__':
    main()