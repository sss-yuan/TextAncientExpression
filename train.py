import os

import numpy as np
import torch
from vocab_utils import build_vocab

from model import TextToAncient
from loader import load_data
from tqdm.auto import tqdm


def train():
    max_length = 80
    batch_size = 1024
    EPOCH = 300
    lr = 1e-4
    log_dir = "log/"
    os.makedirs(log_dir, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    vocab = build_vocab()

    model = TextToAncient(max_length, len(vocab))
    model = model.to(device)
    optimer = torch.optim.Adam(model.parameters(), lr=lr)
    if os.path.exists(log_dir + "init.pth"):
        model.load_state_dict(torch.load(log_dir + 'init.pth'))
        print(f"load state dict from {log_dir}init.pth")

    data_loader = load_data("data/train.tgt", "data/train.src", max_length=max_length, tokenizer=None, batch_size=batch_size)
    test_data_loader = load_data("data/test.tgt", "data/test.src", max_length=max_length, tokenizer=None, batch_size=batch_size)

    for epoch in range(EPOCH):
        process_bar = tqdm(total=len(data_loader))
        process_bar.set_description(f"EPOCH {epoch+1}/{EPOCH}")
        losses = []
        model.train()
        for x1, x2, y in data_loader:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)

            optimer.zero_grad()
            loss = model(x1, x2, y)
            losses.append(loss.detach().item())
            loss.backward()
            optimer.step()
            log = {"loss": loss.detach().item()}
            process_bar.update(1)
            process_bar.set_postfix(**log)
        avg_loss = np.mean(losses)
        print(f"avg loss: {avg_loss}")
        torch.save(model.state_dict(), log_dir + f"epoch_{epoch}_loss_{avg_loss}.pth")
        evaluate(model, epoch, test_data_loader)


def evaluate(model, epoch, test_data_loader):
    # model.eval()
    # with torch.no_grad():
    #     pass
    pass


if __name__ == "__main__":
    train()