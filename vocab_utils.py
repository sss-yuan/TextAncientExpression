

UNK_TOKEN = 1
SOS_token = 2
EOS_token = 3


def build_vocab():
    path_list = ['data/train.src', 'data/train.tgt', 'data/test.src', 'data/test.tgt']
    vocab = {}
    for path in path_list:
        with open(path, "rb") as f:
            data = f.readlines()
        for sentence in data:
            sentence = sentence.decode("utf-8").strip()
            for word in sentence:
                word = word.strip()
                if word and word not in vocab:
                    vocab[word] = len(vocab) + 4
    vocab["<UNK>"] = UNK_TOKEN
    vocab["<SOS>"] = SOS_token
    vocab["<EOS>"] = EOS_token
    return vocab


if __name__ == "__main__":
    build_vocab()