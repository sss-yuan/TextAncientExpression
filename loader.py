import random

import torch
from torch.utils.data import DataLoader, Dataset
from vocab_utils import build_vocab, SOS_token, EOS_token, UNK_TOKEN


class DataGenerator(Dataset):
    def __init__(self, path_modern, path_ancient, max_length, tokenizer):
        with open(path_modern, "rb") as f:
            data_modern = f.readlines()
        with open(path_ancient, "rb") as f:
            data_ancient = f.readlines()
        self.data_modern = [e.decode("utf-8").strip() for e in data_modern]
        self.data_ancient = [e.decode("utf-8").strip() for e in data_ancient]
        self.vocab = build_vocab()
        self.max_length = max_length
        self.tokenizer = tokenizer

    def padding(self, sentence):
        sentence = sentence + [0] * (self.max_length - len(sentence))
        sentence = sentence[:self.max_length]
        return sentence

    def __getitem__(self, item):
        x = self.data_modern[item]
        x = [self.vocab.get(e, UNK_TOKEN) for e in x if e.strip()]
        x = self.padding(x)
        y = self.data_ancient[item]
        y = [SOS_token] + [self.vocab.get(e, UNK_TOKEN) for e in y if e.strip()] + [EOS_token]

        target_index = random.randint(1, len(y) - 1)
        x2 = self.padding(y[:target_index])
        y = y[target_index]

        return torch.LongTensor(x), torch.LongTensor(x2), torch.LongTensor([y])

    # def __getitem__(self, item):
    #     x = self.data_modern[item]
    #     x = self.tokenizer.encode(x, padding="max_length", max_length=self.max_length)
    #     y = self.data_ancient[item]
    #     y = self.tokenizer.encode(y, padding="max_length", max_length=self.max_length)
    #     return torch.LongTensor(x), torch.LongTensor(y)

    def __len__(self):
        assert len(self.data_modern) == len(self.data_ancient)
        return len(self.data_modern)


def load_data(path_modern, path_ancient, max_length, tokenizer, batch_size):
    dataGenerator = DataGenerator(path_modern, path_ancient, max_length, tokenizer)
    dataLoader = DataLoader(dataGenerator, batch_size=batch_size, shuffle=True)
    return dataLoader
