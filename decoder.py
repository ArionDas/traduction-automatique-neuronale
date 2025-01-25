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

from config import CONFIG
from attention import BahdanauAttention

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AttnDecoderRNN(nn.Module):
    
    def __init__(self, hidden_size, output_size, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.GRU = nn.GRU(2 * self.hidden_size, self.hidden_size, batch_first=True)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.dropout = nn.Dropout(self.dropout_p)
        
        self.attention = BahdanauAttention(self.hidden_size)
    
     
    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        
        batch_size = encoder_outputs.size(0)
        
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(CONFIG["SOS_TOKEN"])
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []
        
        for i in range(CONFIG["MAX_LENGTH"]):
            decoder_output, decoder_hidden, attention = self.forward_step(
                decoder_input,
                decoder_hidden,
                encoder_outputs
            )
            
            decoder_outputs.append(decoder_output)
            attentions.append(attention)
            
            if target_tensor is not None:
                decoder_input = target_tensor[:, i].unsqueeze(1) ## teacher forcing
            else:
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()
                
        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)
        
        return decoder_outputs, decoder_hidden, attentions
        
        
    def forward_step(self, input, hidden, encoder_outputs):
        
        embedded = self.embedding(input)
        embedded = self.dropout(embedded)
        
        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat([embedded, context], dim=2)
        
        output, hidden = self.GRU(input_gru, hidden)
        output = self.out(output)
        
        return output, hidden, attn_weights