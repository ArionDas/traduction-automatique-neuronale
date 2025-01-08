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


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        
        self.wa = nn.Linear(hidden_size, hidden_size)
        self.ua = nn.Linear(hidden_size, hidden_size)
        self.va = nn.Linear(hidden_size, 1)
        
    def forward(self, query, keys):
        scores = self.va(torch.tanh(self.wa(query) + self.ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)
        
        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)
        
        return context, weights