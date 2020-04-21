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

    arg.data =
