
import torch

from model import TextToAncient
from vocab_utils import build_vocab, UNK_TOKEN, SOS_token, EOS_token

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

max_length = 80
vocab = build_vocab()
index_to_word = {}
for k, v in vocab.items():
    index_to_word[v] = k

model = TextToAncient(max_length, len(vocab))
model = model.to(device)
model.eval()
model.load_state_dict(torch.load('log/final_weights.pth'))


while True:
    x = input("输入（例：梁惠王站在池塘边上，一面顾盼着鸿雁麋鹿，一面说：贤人也以此为乐吗？）：")
    x = [vocab.get(e, UNK_TOKEN) for e in x if e.strip()]
    x = x[:max_length]
    x += [0] * (max_length - len(x))
    x = torch.LongTensor([x])
    x = x.to(device)

    x2_begin = [SOS_token]
    x2 = x2_begin[:max_length]
    x2 += [0] * (max_length - len(x2))
    x2 = torch.LongTensor([x2])
    x2 = x2.to(device)

    y = model(x, x2)
    y = torch.argmax(y, dim=-1)
    y = y.item()
    res = [index_to_word.get(y, index_to_word[UNK_TOKEN])]

    while y != EOS_token:
        x2_begin.append(y)
        x2 = x2_begin[:max_length]
        x2 += [0] * (max_length - len(x2))
        x2 = torch.LongTensor([x2])
        x2 = x2.to(device)

        y = model(x, x2)
        y = torch.argmax(y, dim=-1)
        y = y.item()
        if y != EOS_token:
            res.append(index_to_word.get(y, index_to_word[UNK_TOKEN]))

    print("".join(res))