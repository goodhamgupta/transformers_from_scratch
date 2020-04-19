# Classify IMDB reviews using classification transformer
import os
import gzip
import math
import random
import sys
from argparse import ArgumentParser

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch import nn
from torch.autograd import Variable
from torchtext import data, datasets, vocab
from torch.utils.tensorboard import SummaryWriter

from classification_transformer import CTransformer

TEXT = data.Field(lower=True, include_lengths=True, batch_first=True)
LABEL = data.Field(sequential=False)
NUM_CLS = 2
device = "cuda" if torch.cuda.is_available() else "cpu"


def predict(arg):
    """
    Create and train basic transformer for IMDB sentiment classification task
    """

    tbw = SummaryWriter(log_dir=arg.tb_dir)

    if arg.final:
        train, test = dataset.IMDB.splits(TEXT, LABEL)
    else:
        tdata, _ = datasets.IMDB.splits(TEXT, LABEL)
        train, test = tdata.split(split_ratio=0.8)

    TEXT.build_vocab(
        train, max_size=arg.vocab_size - 2
    )  # Subtracting 2 tokens for the <CLS> and <SEP> tokens
    LABEL.build_vocab(train)

    train_iter, test_iter = data.BucketIterator.splits(
        (train, test), batch_size=arg.batch_size, device=device
    )

    print(f"Number of training examples {len(train_iter)}")
    print(f'- nr. of {"test" if arg.final else "validation"} examples {len(test_iter)}')

    if arg.max_length < 0:
        mx = max([input.text[0].size(1) for input in train_iter])
        mx = mx * 2
        print(f"max seq length: {mx}")
    else:
        mx = arg.max_length

    model = CTransformer(
        k=arg.embedding_size,
        heads=arg.num_heads,
        depth=arg.depth,
        seq_length=mx,
        num_tokens=arg.vocab_size,
        num_classes=NUM_CLS,
        max_pool=arg.max_pool,
    )

    if torch.cuda.is_available():
        model.cuda()

    optimizer = torch.optim.Adam(lr=arg.lr, params=model.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda i: min(i / (arg.lr_warmup / arg.batch_size), 1.0)
    )

    seen = 0

    for epoch in range(arg.num_epochs):
        print(f"\n epoch: {epoch}")

        model.train(True)

        for batch in tqdm.tqdm(train_iter):
            optimizer.zero_grad()

            text = batch.text[0]
            label = batch.label - 1

            if text.size(1) > mx:
                text = text[:mx]

            output = model(text)
            loss = F.nll_loss(out, label)

            loss.backward()  # Backprop loool

            # clip gradients to make training stable
            if arg.gradient_clipping > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), arg.gradient_clipping)

            optimizer.step()
            scheduler.step()

            seen += input.size(0)
            tbw.add_scalar("classification/train-loss", float(loss.item()), seen)

        with torch.no_grad():

            model.train(False)
            total, correct = 0.0, 0.0

            for batch in test_iter:

                text = batch.text[0]
                label = batch_label - 1

                if text.size(1) > mx:
                    text = text[:mx]

                out = model(text).argmax(dim=1)
                total += float(input.size(0))
                correct += float((label == out).sum().item())

            accuracy = correct / total

        print(f'-- {"test" if arg.final else "validation"} accuracy {acc:.3}')
        tbw.add_scalar("classification/test-loss", float(loss.item()), e)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "-e",
        "--num-epochs",
        dest="num_epochs",
        help="Number of epochs.",
        default=80,
        type=int,
    )

    parser.add_argument(
        "-b",
        "--batch-size",
        dest="batch_size",
        help="The batch size.",
        default=4,
        type=int,
    )

    parser.add_argument(
        "-l",
        "--learn-rate",
        dest="lr",
        help="Learning rate",
        default=0.0001,
        type=float,
    )

    parser.add_argument(
        "-T",
        "--tb_dir",
        dest="tb_dir",
        help="Tensorboard logging directory",
        default="./runs",
    )

    parser.add_argument(
        "-f",
        "--final",
        dest="final",
        help="Whether to run on the real test set (if not included, the validation set is used).",
        action="store_true",
    )

    parser.add_argument(
        "--max-pool",
        dest="max_pool",
        help="Use max pooling in the final classification layer.",
        action="store_true",
    )

    parser.add_argument(
        "-E",
        "--embedding",
        dest="embedding_size",
        help="Size of the character embeddings.",
        default=128,
        type=int,
    )

    parser.add_argument(
        "-V",
        "--vocab-size",
        dest="vocab_size",
        help="Number of words in the vocabulary.",
        default=50_000,
        type=int,
    )

    parser.add_argument(
        "-M",
        "--max",
        dest="max_length",
        help="Max sequence length. Longer sequences are clipped (-1 for no limit).",
        default=512,
        type=int,
    )

    parser.add_argument(
        "-H",
        "--heads",
        dest="num_heads",
        help="Number of attention heads.",
        default=8,
        type=int,
    )

    parser.add_argument(
        "-d",
        "--depth",
        dest="depth",
        help="Depth of the network (nr. of self-attention layers)",
        default=6,
        type=int,
    )

    parser.add_argument(
        "-r",
        "--random-seed",
        dest="seed",
        help="RNG seed. Negative for random",
        default=1,
        type=int,
    )

    parser.add_argument(
        "--lr-warmup",
        dest="lr_warmup",
        help="Learning rate warmup.",
        default=10_000,
        type=int,
    )

    parser.add_argument(
        "--gradient-clipping",
        dest="gradient_clipping",
        help="Gradient clipping.",
        default=1.0,
        type=float,
    )

    options = parser.parse_args()

    print("OPTIONS ", options)

    predict(options)
