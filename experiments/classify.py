# Classify IMDB reviews using classification transformer

import gzip
import math
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch import nn
from torch.autograd import Variable
from torchtext import data, datasts, vocab
