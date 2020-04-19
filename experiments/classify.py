# Classify IMDB reviews using classification transformer

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

from torchtext import data, datasts, vocab
import random, tqdm, sys, math, gzip
