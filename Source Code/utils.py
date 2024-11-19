import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from tokenizers import Tokenizer, WordLevel, Whitespace, WordLevelTrainer, TemplateProcessing
import math
from encoder import EncoderLayer
from decoder import DecoderLayer

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()    
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, q, k, v, mask=None):
        batch_size, seq_len, _ = q.size()
        
        qkv = self.qkv_proj(torch.cat([q, k, v], dim=0)).chunk(3, dim=0)
        q, k, v = [x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2) for x in qkv]
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention = F.softmax(scores, dim=-1)
        context = torch.matmul(attention, v)
        
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.output_proj(context)

class PositionWiseFeedForward(nn.Module):
    def __init__(self, dim_model, dim_feedforward):
        super().__init__()
        self.fc1 = nn.Linear(dim_model, dim_feedforward)
        self.fc2 = nn.Linear(dim_feedforward, dim_model)
    
    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_length):
        super().__init__()
        position = torch.arange(max_length).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim))
        encoding = torch.zeros(1, max_length, embed_dim)
        encoding[0, :, 0::2] = torch.sin(position * div_term)
        encoding[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('encoding', encoding)
    
    def forward(self, x):
        return x + self.encoding[:, :x.size(1)]

class Transformer(nn.Module):
    def __init__(self, vocab_sizes, dim_model, num_heads, num_layers, dim_feedforward, max_seq_length, dropout_rate, device):
        super().__init__()
        self.device = device
        self.embeddings = nn.ModuleDict({
            'src': nn.Embedding(vocab_sizes['src'], dim_model),
            'tgt': nn.Embedding(vocab_sizes['tgt'], dim_model)
        })
        self.pos_encoder = PositionalEncoding(dim_model, max_seq_length)
        self.encoder = nn.ModuleList([EncoderLayer(dim_model, num_heads, dim_feedforward, dropout_rate) for _ in range(num_layers)])
        self.decoder = nn.ModuleList([DecoderLayer(dim_model, num_heads, dim_feedforward, dropout_rate) for _ in range(num_layers)])
        self.final_proj = nn.Linear(dim_model, vocab_sizes['tgt'])
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, src, tgt):
        # Create masks
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        causal_mask = torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1).bool()
        tgt_mask = tgt_mask & ~causal_mask.to(self.device)
        src_mask, tgt_mask = src_mask.to(self.device), tgt_mask.to(self.device)

        # Encoding
        src_embed = self.dropout(self.pos_encoder(self.embeddings['src'](src)))
        enc_output = src_embed
        for layer in self.encoder:
            enc_output = layer(enc_output, src_mask)

        # Decoding
        tgt_embed = self.dropout(self.pos_encoder(self.embeddings['tgt'](tgt)))
        dec_output = tgt_embed
        for layer in self.decoder:
            dec_output = layer(dec_output, enc_output, src_mask, tgt_mask)

        return self.final_proj(dec_output)

def train_tokenizer(english_file, french_file):
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[START]", "[END]"])

    def batch_iterator(file_paths):
        for file_path in file_paths:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    yield line.strip()

    tokenizer.train_from_iterator(batch_iterator([english_file, french_file]), trainer=trainer)

    tokenizer.post_processor = TemplateProcessing(
        single="[START] $A [END]",
        pair="[START] $A [END] $B:1 [END]:1",
        special_tokens=[
            ("[START]", tokenizer.token_to_id("[START]")),
            ("[END]", tokenizer.token_to_id("[END]")),
        ],
    )

    return tokenizer

class TranslationDataset(Dataset):
    def __init__(self, src_file, tgt_file, tokenizer, max_length=50):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self.load_data(src_file, tgt_file)
    
    def load_data(self, src_file, tgt_file):
        data = []
        with open(src_file, 'r', encoding='utf-8') as src_f, open(tgt_file, 'r', encoding='utf-8') as tgt_f:
            for src_line, tgt_line in zip(src_f, tgt_f):
                src_encoded = self.tokenizer.encode(src_line.strip()).ids[:self.max_length]
                tgt_encoded = self.tokenizer.encode(tgt_line.strip()).ids[:self.max_length]
                data.append((src_encoded, tgt_encoded))
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch, pad_id, max_length):
    src_batch, tgt_batch = zip(*batch)
    src_batch = [src + [pad_id] * (max_length - len(src)) for src in src_batch]
    tgt_batch = [tgt + [pad_id] * (max_length - len(tgt)) for tgt in tgt_batch]
    return torch.tensor(src_batch), torch.tensor(tgt_batch)