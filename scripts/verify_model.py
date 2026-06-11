import os
import torch
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from arthaml.models.encoder import Encoder
from arthaml.models.decoder import Attention, Decoder
from arthaml.models.seq2seq import Seq2Seq

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    encoder = Encoder(vocab_size=21551, embed_dim=256, hidden_size=512, num_layers=2, dropout_p=0.3, padding_idx=0)
    attention = Attention(hidden_size=512)
    decoder = Decoder(attention, vocab_size=26802, embed_dim=256, hidden_size=512, num_layers=2, dropout_p=0.3, padding_idx=0)
    
    model = Seq2Seq(encoder, decoder, device).to(device)
    
    src = torch.randint(0, 21551, (32, 50)).to(device)
    src_len = torch.full((32,), 50).to(device)
    tgt = torch.randint(0, 26802, (32, 50)).to(device)
    
    print("Running forward pass...")
    outputs = model(src, src_len, tgt, teacher_forcing_ratio=0.5)
    
    print(f"Expected output shape: [32, 50, 26802]")
    print(f"Actual output shape:   {list(outputs.shape)}")
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")

if __name__ == "__main__":
    main()
