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

class DecoderLayer(nn.Module):
    def __init__(self, model_dimension, num_attention_heads, feedforward_dimension, dropout_rate):
        super().__init__()
        self.self_attention = MultiHeadAttention(model_dimension, num_attention_heads)
        self.cross_attention = MultiHeadAttention(model_dimension, num_attention_heads)
        self.feedforward = PositionWiseFeedForward(model_dimension, feedforward_dimension)
        
        self.layer_norms = nn.ModuleList([nn.LayerNorm(model_dimension) for _ in range(3)])
        self.dropout_rate = dropout_rate

    def apply_attention(self, query, key_value, attention_mask, layer_norm_index):
        attention_output = self.self_attention(query, key_value, key_value, attention_mask)
        return self.add_and_norm(query, attention_output, layer_norm_index)

    def add_and_norm(self, residual, sublayer_output, layer_norm_index):
        dropout_output = F.dropout(sublayer_output, p=self.dropout_rate, training=self.training)
        return self.layer_norms[layer_norm_index](residual + dropout_output)

    def forward(self, decoder_input, encoder_output, source_mask, target_mask):
        # Self-attention
        self_attention_output = self.apply_attention(decoder_input, decoder_input, target_mask, 0)
        
        # Cross-attention
        cross_attention_output = self.apply_attention(self_attention_output, encoder_output, source_mask, 1)
        
        # Feedforward
        feedforward_output = self.feedforward(cross_attention_output)
        decoder_output = self.add_and_norm(cross_attention_output, feedforward_output, 2)
        
        return decoder_output