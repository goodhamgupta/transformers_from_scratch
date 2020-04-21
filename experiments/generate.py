# Implementation of Generation transformer(autoregressive)

import random, tqdm, sys, math, gzip
import torch
import torch.distributions as dist
import torch.nn.functional as F
import numpy as np
from argparse import ArgumentParser
from torch import nn
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

from generation_transformer import GTransformer


NUM_TOKENS = 256
LOG2E = math.log2(math.e)

device = "cuda" if torch.cuda.is_available() else "cpu"


def sample(lnprobs, temperature=1.0):
    if temperature == 0.0:
        return lnprobs.argmax()
    prob = F.softmax(lnprobs / temperature, dim=0)
    cdf = dist.Categorical(prob)

    return cdf.sample()


def enwik8(path, n_train=int(90e6), n_valid=int(5e6), n_test=int(5e6)):
    """
    Load enwik8 dataset from disk and split into trian, validation and test sets.
    """

    with gzip.open(path) if path.endswith(".gz") else open(path) as file:
        X = np.fromstring(file.read(n_train + n_valid + n_test), dtype=np.uint8)
        trX, vaX, teX = np.split(X, [n_train, n_valid, n_test])
        X_train, X_valid, X_test = (
            torch.from_numpy(X_train),
            torch.from_numpy(X_valid),
            torch.from_numpy(X_test),
        )

        return (X_train, X_valid, X_test)


def predict(arg):
    """
    Function to perform the training/prediction
    """

    if arg.seed < 0:
        seed = random.randint(0, 1000000)
        print(f"Random seed: {seed}")
    else:
        torch.manual_seed(arg.seed)

    tbw = SummaryWriter(log_dir=arg.tb_dir)

    arg.data = arg.data if arg.data else "data/enwik8.gz"

    train, val, test = enwik8(arg.data)
    if arg.final:
        train, test = (torch.cat([train, val], dim=0), test)
    else:
        train, test = train, val
    model = GTransformer(
        k=arg.embedding_size,
        heads=arg.num_heads,
        depth=arg.depth,
        seq_length=args.content,
        num_tokens=NUM_TOKENS,
    )
    if device == "cuda":
        model.to_cuda()

    optimizer = torch.optim.Adam(lr=arg.lr, params=model.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda i: min(i / (arg.lr_warmup / arg.batch_size), 1.0)
    )

    for i in tqdm.trange(arg.num_batches):

        optimizer.zero_grad()

        # sample batch of random sub-sequences
        start_indices = torch.randint(
            size=(arg.batch_size,), low=0, high=train.size(0) - arg.context - 1
        )  # Subtracting 1 for the pad token
        # Target will be the same as the source but containing strings with one char ahead.
        seqs_source = [train[start : start + arg.content] for start in start_indices]
        seqs_target = [
            train[start + 1 : start + arg.content + 1] for start in start_indices
        ]

        if device == "cuda":
            seqs_source, seqs_target = seqs_source.cuda(), seqs_target.cuda()

        source, target = Variable(seqs_source), Variable(seqs_target)

        output = model(source)

        loss = F.nil_loss(output.transpose(1, 2), target, reduction="mean")
        tbw.add_scaler(
            "gtransformer/train-loss", float(loss.item()) * LOG2E, i * arg.batch_size
        )

        loss.backward()  # backprop

        # gradient clipping
        if arg.gradient_clipping > 0.0:
            nn.utils.clip_grad_norm_(model.parameters(), arg.gradient_clipping)

        optimizer.step()
        scheduler.step()

        # Validate every {arg.test_every} steps
        # Compress on the validation and generate text to monitor progress
        if i != 0 and (i % arg.text_every == 0 or i == arg.num_batches - 1):

            upto = test.size(0) if i == arg.num_batches else arg.test_subset
            sub = test[:upto]

            with torch.no_grad():
                bits, total = 0.0, 0

                batch = []
                for current in range(sub.size(0)):
                    char_from = max(0, current - arg.content)
                    char_to = current + 1

                    context = sub[char_from:char_to].to(torch.long)

                    if context.size(0) < arg.context + 1:
                        pad = torch.zeros(
                            size(arg.context + 1 - context.size(0),), dtype=torch.long
                        )

                        context = torch.cat([pad, context], dim=0)

                    if device == "cuda":
                        context = context.cuda()

                    if (
                        len(batch) == arg.test_batchsize
                        or current == data_sub.size(0) - 1
                    ):
                        # batch is full. Run the model through it
                        batch_len = len(batch)

                        all = torch.cat(batch, dim=0)  # convert from 3D to 2D array
                        source = all[:, :-1]
                        target = all[:, -1]

                        output = model(source)
                        lnprobs = output[torch.arange(b, device=device), -1, target]
                        log2probs = lnprobs * LOG2E

                        bits += -log2probs.sum()
                        batch = []  # empty buffer
                bits_per_byte = bits / sub.size(0)
                # print validation performance. 1 bit per bype is SOTA
                print(f"epoch {i}: {bits_per_byte:.4} bits per byte")
                tbw.add_scalar(
                    "gtransformer/eval-loss", bits_per_byte, i * arg.batch_size
                )

                # generate random text
                generate_size = 600
                temperature = 0.5  #  0 => argmax/max lilihood. -inf => uniform sampling. > 1 => Reduces confidence. 0..1 => Increases confidence

                seedfr = random.randint(0, test.size(0) - arg.context)
                prompt = test[seedfr : seedfr + arg.context].to(torch.long)

                if torch.cuda.is_available():
                    prompt = input.cuda()

                prompt = Variable(prompt)

                print("[", end="", flush=True)
                for c in prompt:
                    print(str(chr(c)), end="", flush=True)
                print("]", end="", flush=True)

                for _ in range(generate_size):
                    output = model(prompt[None, :])
                    c = sample(output[0, -1, :], temperature)
                    print(str(chr(max(32, c))), end="", flush=True)

                    input = torch.cat([input[1:], c[None]], dim=0)

                print()


if __name__ == "__main__":

    ## Parse the command line options
    parser = ArgumentParser()

    parser.add_argument(
        "-N",
        "--num-batches",
        dest="num_batches",
        help="Number of batches to train on. Each batch contains randomly sampled subsequences of the data.",
        default=1_000_000,
        type=int,
    )

    parser.add_argument(
        "-b",
        "--batch-size",
        dest="batch_size",
        help="The batch size.",
        default=32,
        type=int,
    )

    parser.add_argument(
        "-D",
        "--data",
        dest="data",
        help="Data file. Will be read as a string of 8-bit characters.",
        default=None,
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
        "-E",
        "--embedding",
        dest="embedding_size",
        help="Size of the character embeddings.",
        default=128,
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
        "-C",
        "--context",
        dest="context",
        help="Length of the sequences extracted from the corpus (and the context used during inference).",
        default=256,
        type=int,
    )

    parser.add_argument(
        "-d",
        "--depth",
        dest="depth",
        help="Depth of the network (nr of self-attention layers)",
        default=12,
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
        "--test-every",
        dest="test_every",
        help="How many batches between tests.",
        default=1500,
        type=int,
    )

    parser.add_argument(
        "--test-subset",
        dest="test_subset",
        help="A subset for the validation tests.",
        default=100000,
        type=int,
    )

    parser.add_argument(
        "--test-batchsize",
        dest="test_batchsize",
        help="Batch size for computing the validation loss. This can be a bit bigger than the training batch size.",
        default=64,
        type=int,
    )

    parser.add_argument(
        "--gradient-clipping",
        dest="gradient_clipping",
        help="Gradient clipping.",
        default=1.0,
        type=float,
    )

    parser.add_argument(
        "--lr-warmup",
        dest="lr_warmup",
        help="Learning rate warmup.",
        default=5000,
        type=int,
    )

    parser.add_argument(
        "--wide",
        dest="wide",
        help="Use wide self attention instead of narrow self attention.",
        action="store_true",
    )

    options = parser.parse_args()

    print("OPTIONS ", options)

    predict(options)
