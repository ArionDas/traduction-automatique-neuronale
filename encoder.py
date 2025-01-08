from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EncoderRNN(nn.Module):
    def __init__(
        self, input_size, hidden_size, dropout_p=0.1 
    ):
        super(EncoderRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.GRU = nn.GRU(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        
    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.GRU(embedded)
        return output, hidden