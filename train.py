from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import re
import random
import time
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from config import CONFIG
from encoder import EncoderRNN
from decoder import AttnDecoderRNN
from data_preparation import get_dataloader


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def train_epoch(dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    
    total_loss = 0
    for data in dataloader:
        input_tensor, target_tensor = data
        
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        
        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)
        
        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            target_tensor.view(-1)
        )
        loss.backward()
        
        encoder_optimizer.step()
        decoder_optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader)


def train(train_dataloader, encoder, decoder, n_epochs, lr=CONFIG["lr"], print_every=100, plot_every=100):
    
    start = time.time()
    plot_losses = []
    print_loss_total = 0 ## Reset every print_every
    plot_loss_total = 0 ## Reset every plot_every
    
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=lr)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=lr)
    criterion = nn.NLLLoss()
    
    for epoch in range(1, n_epochs + 1):
        
        loss = train_epoch(train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss
        
        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, epoch / n_epochs), epoch, epoch / n_epochs * 100, print_loss_avg))
            
        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
            
    showPlot(plot_losses)
    


plt.switch_backend('agg')
def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    
    
def main():
    hidden_size = CONFIG["hidden_size"]
    batch_size = CONFIG["batch_size"]
    
    input_lang, output_lang, train_dataloader = get_dataloader(batch_size)
    
    encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    decoder = AttnDecoderRNN(hidden_size, output_lang.n_words).to(device)
    
    train(train_dataloader, encoder, decoder, CONFIG["n_epochs"])
    

if __name__ == "__main__":
    main()