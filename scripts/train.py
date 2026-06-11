import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from arthaml.data.vocabulary import Vocabulary
from arthaml.data.dataset import TranslationDataset
from arthaml.models.encoder import Encoder
from arthaml.models.decoder import Attention, Decoder
from arthaml.models.seq2seq import Seq2Seq
from arthaml.evaluation.bleu import compute_bleu

def train(model, iterator, optimizer, criterion, clip, device):
    model.train()
    epoch_loss = 0
    
    for i, batch in enumerate(iterator):
        src = batch['src'].to(device)
        tgt = batch['tgt'].to(device)
        src_len = batch['src_len'].to(device)
        
        optimizer.zero_grad()
        
        output = model(src, src_len, tgt, teacher_forcing_ratio=0.5)
        
        output_dim = output.shape[-1]
        
        # Ignored t=0 since it's the <SOS> token
        output = output[:, 1:].reshape(-1, output_dim)
        tgt = tgt[:, 1:].reshape(-1)
        
        loss = criterion(output, tgt)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion, device):
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch['src'].to(device)
            tgt = batch['tgt'].to(device)
            src_len = batch['src_len'].to(device)
            
            output = model(src, src_len, tgt, teacher_forcing_ratio=0)
            
            output_dim = output.shape[-1]
            
            output = output[:, 1:].reshape(-1, output_dim)
            tgt = tgt[:, 1:].reshape(-1)
            
            loss = criterion(output, tgt)
            epoch_loss += loss.item()
            
    return epoch_loss / len(iterator)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs(os.path.join(project_root, 'models'), exist_ok=True)
    
    print("Loading vocabularies...")
    vocab_en = Vocabulary("English")
    vocab_en.load(os.path.join(project_root, 'data/processed/vocab_en.json'))
    vocab_kn = Vocabulary("Kannada")
    vocab_kn.load(os.path.join(project_root, 'data/processed/vocab_kn.json'))
    
    print("Loading dataset...")
    data_path = os.path.join(project_root, 'data/processed/clean_50k.json')
    dataset = TranslationDataset(data_path, vocab_en, vocab_kn, max_len=50)
    
    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_iterator = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_iterator = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    print("Initializing models...")
    encoder = Encoder(vocab_size=21551, embed_dim=256, hidden_size=512, num_layers=2, dropout_p=0.3, padding_idx=0)
    attention = Attention(hidden_size=512)
    decoder = Decoder(attention, vocab_size=26802, embed_dim=256, hidden_size=512, num_layers=2, dropout_p=0.3, padding_idx=0)
    
    model = Seq2Seq(encoder, decoder, device).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    N_EPOCHS = 10
    CLIP = 1.0
    
    best_val_loss = float('inf')
    best_bleu = 0.0
    
    print("Starting training...")
    for epoch in range(N_EPOCHS):
        start_time = time.time()
        
        train_loss = train(model, train_iterator, optimizer, criterion, CLIP, device)
        val_loss = evaluate(model, val_iterator, criterion, device)
        
        bleu_score = compute_bleu(model, val_dataset, vocab_kn, device, n=500)
        
        end_time = time.time()
        time_taken = int(end_time - start_time)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(project_root, 'models/best_model.pth'))
            
        if bleu_score > best_bleu:
            best_bleu = bleu_score
            
        print(f"Epoch {epoch+1:02} | Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f} | BLEU: {bleu_score:.2f} | Time: {time_taken}s")
        
    torch.save(model.state_dict(), os.path.join(project_root, 'models/final_model.pth'))
    print(f"\nTraining Complete!")
    print(f"Best Val Loss: {best_val_loss:.3f}")
    print(f"Best BLEU Score: {best_bleu:.2f}")

if __name__ == "__main__":
    main()
