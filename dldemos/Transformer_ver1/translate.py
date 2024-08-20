import torch

from dldemos.Transformer_ver1.dataset import MAX_SEQ_LEN, tensor_to_sentence
from dldemos.Transformer_ver1.model import Transformer
from dldemos.Transformer_ver1.preprocess_data import (EOS_ID, PAD_ID,
                                                      SOS_ID,
                                                      get_dataloader,
                                                      load_sentences,
                                                      load_vocab)

# Config
batch_size = 64
lr = 0.0001
d_model = 512
d_ff = 2048
n_layers = 6
heads = 8
dropout_rate = 0.2
max_len = 50

def main():
    model_path = 'model-e10.pth'

    device = 'cuda'
    en_vocab, zh_vocab = load_vocab()
    word2idx = {word: idx for idx, word in enumerate(en_vocab)}

    en_train, zh_train, en_valid, zh_valid = load_sentences()
    dataloader_valid = get_dataloader(en_train, zh_train, 1)

    model = Transformer(len(en_vocab),
                        len(zh_vocab),
                        PAD_ID,
                        d_model,
                        d_ff,
                        n_layers,
                        heads,
                        dropout_rate,
                        max_seq_len=MAX_SEQ_LEN)
    model.to(device)
    model.load_state_dict(torch.load(model_path))

    my_input = ['we', 'should', 'protect', 'environment']
    x_batch = torch.LongTensor([[word2idx[x] for x in my_input]]).to(device)

    cn_sentence = tensor_to_sentence(x_batch[0], en_vocab, True)
    print(cn_sentence)

    y_input = torch.ones(batch_size, max_len,
                         dtype=torch.long).to(device) * PAD_ID
    y_input[0] = SOS_ID #word2idx['<SOS>']

    cnt = 0
    with torch.no_grad():
        x, y = x_batch.to(device), y_input.to(device)
        x_mask = x == PAD_ID
        n = x.shape[0]
        sample = torch.ones(n, MAX_SEQ_LEN,
                            dtype=torch.long).to(device) * PAD_ID
        sample[:, 0] = SOS_ID
        print(tensor_to_sentence(x[0], en_vocab, True))
        print(tensor_to_sentence(y[0], zh_vocab))
        for i in range(50):
            sample_mask = sample == PAD_ID
            # y_predict = model(x, sample, x_mask, sample_mask)
            y_predict = model(x, sample)

            y_predict = y_predict[:, i]
            prob_dist = torch.softmax(y_predict, 1)
            # new_word = torch.multinomial(prob_dist, 1)
            _, new_word = torch.max(prob_dist, 1)
            sample[:, i + 1] = new_word
            s = (tensor_to_sentence(sample[0], zh_vocab))
            print(s)

            # by dum,提前结束
            lastword = zh_vocab[int(new_word.item())]
            print("lastword ", lastword)
            if lastword == "<eos>":
                break

        # w = tensor_to_sentence(new_word,zh_vocab)
        # if w == "<eos>":
        #     break
        cnt += 1
    print('Done.')


if __name__ == '__main__':
    main()
