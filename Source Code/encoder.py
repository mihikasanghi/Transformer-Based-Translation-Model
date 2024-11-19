import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import math
import copy
import time
from torch.utils.data import Dataset, DataLoader
from utils import MultiHeadAttention, PositionWiseFeedForward


class EncoderLayer(nn.Module):
    def __init__(self, model_dimension, num_attention_heads, feedforward_dimension, dropout_rate):
        super().__init__()
        self.self_attention = MultiHeadAttention(model_dimension, num_attention_heads)
        self.feedforward = PositionWiseFeedForward(model_dimension, feedforward_dimension)
        self.layer_norm1 = nn.LayerNorm(model_dimension)
        self.layer_norm2 = nn.LayerNorm(model_dimension)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, encoder_input, attention_mask):
        # Self-attention
        attention_output = self.self_attention(encoder_input, encoder_input, encoder_input, attention_mask)
        attention_output = self.dropout(attention_output)
        attention_output = self.layer_norm1(encoder_input + attention_output)
        
        # Feedforward
        feedforward_output = self.feedforward(attention_output)
        feedforward_output = self.dropout(feedforward_output)
        encoder_output = self.layer_norm2(attention_output + feedforward_output)
        
        return encoder_output
