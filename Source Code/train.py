import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from utils import Transformer, TranslationDataset, collate_fn, train_tokenizer

def train_and_evaluate(model, train_loader, val_loader, tokenizer, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id("[PAD]"))
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    
    best_val_loss = float('inf')
    
    for epoch in range(config['num_epochs']):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        
        print(f'Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), config['model_save_path'])
            print(f'Best model saved with validation loss: {best_val_loss:.4f}')

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    start_time = time.time()
    
    for batch_idx, (src, tgt) in enumerate(dataloader):
        src, tgt = src.to(device), tgt.to(device)
        tgt_input, tgt_output = tgt[:, :-1], tgt[:, 1:]
        
        optimizer.zero_grad()
        output = model(src, tgt_input)
        loss = criterion(output.contiguous().view(-1, output.size(-1)), tgt_output.contiguous().view(-1))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            current_loss = total_loss / (batch_idx + 1)
            elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
            print(f'Batch {batch_idx}/{len(dataloader)} | Loss: {current_loss:.4f} | Elapsed: {elapsed}')
    
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)
            tgt_input, tgt_output = tgt[:, :-1], tgt[:, 1:]
            
            output = model(src, tgt_input)
            loss = criterion(output.contiguous().view(-1, output.size(-1)), tgt_output.contiguous().view(-1))
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

def main():
    config = {
        'dim_model': 1024,
        'num_heads': 4,
        'num_layers': 3,
        'dim_feedforward': 2048,
        'max_seq_length': 25,
        'dropout_rate': 0.1,
        'batch_size': 128,
        'num_epochs': 20,
        'model_save_path': 'transformer.pth'
    }

    tokenizer = train_tokenizer("ted-talks-corpus/train.en", "ted-talks-corpus/train.fr")
    pad_id = tokenizer.token_to_id("[PAD]")

    train_dataset = TranslationDataset("ted-talks-corpus/train.en", "ted-talks-corpus/train.fr", tokenizer)
    val_dataset = TranslationDataset("ted-talks-corpus/dev.en", "ted-talks-corpus/dev.fr", tokenizer)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        collate_fn=lambda b: collate_fn(b, pad_id, config['max_seq_length']),
        num_workers=4
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        collate_fn=lambda b: collate_fn(b, pad_id, config['max_seq_length']),
        num_workers=4
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Transformer(
        src_vocab_size=tokenizer.get_vocab_size(),
        tgt_vocab_size=tokenizer.get_vocab_size(),
        d_model=1024,
        num_heads=4,
        num_layers=3,
        d_ff=2048,
        max_seq_length=25,
        dropout=0.1,
        device=device
    ).to(device)

    model.load_state_dict(torch.load('transformer.pth'))
    
    train_and_evaluate(model, train_loader, val_loader, tokenizer, config)

if __name__ == '__main__':
    main()