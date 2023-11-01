import torch
import torch.nn as nn
import torch.nn.functional as F


class TextToAncient(nn.Module):
    def __init__(self, max_length, vocab_size):
        super(TextToAncient, self).__init__()

        self.hidden_size = 512

        self.max_length = max_length

        # self.bert = BertModel.from_pretrained(bert_base_path, return_dict=False)
        # if not bert_train:
        #     for p in self.bert.parameters():
        #         p.requires_grad = False

        self.embedding = nn.Embedding(vocab_size + 1, self.hidden_size, padding_idx=0)
        self.lstm1 = nn.LSTM(self.hidden_size, self.hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(self.hidden_size, self.hidden_size, batch_first=True)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)

        self.linear1 = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, vocab_size)

    def forward(self, x1, x2, y=None):

        x1 = self.embedding(x1)
        x1, (h1, _) = self.lstm1(x1)
        h1 = h1.squeeze(1)

        x2 = self.embedding(x2)
        _, (h2, _) = self.lstm2(x2)
        h2 = h2.squeeze(1)

        attn = F.softmax(self.attn(torch.cat([h1, h2], dim=-1)), dim=-1)
        attn = attn.unsqueeze(1)
        x1 = torch.bmm(attn, x1)
        x1 = x1.squeeze(1)

        out = self.linear2(F.gelu(self.linear1(torch.cat([x1, h2], dim=-1))))
        if y is not None:
            return F.cross_entropy(out, y.reshape(-1), ignore_index=0)
        else:
            return F.softmax(out, dim=-1)


if __name__ == "__main__":
    from torchinfo import summary
    from vocab_utils import build_vocab
    # model = TextToAncient("C:\\yuanzhouli\\AllCodeRelated\\bert\\bert-base-chinese", 80)
    model = TextToAncient(80, len(build_vocab()))
    summary(model)
