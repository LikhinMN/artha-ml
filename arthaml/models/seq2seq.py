import torch
import torch.nn as nn
import random

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, src_len, tgt, teacher_forcing_ratio=0.5):
        # src: [batch_size, src_len]
        # src_len: [batch_size]
        # tgt: [batch_size, tgt_len]
        
        batch_size = src.shape[0]
        tgt_len = tgt.shape[1]
        tgt_vocab_size = self.decoder.vocab_size
        
        # Tensor to store decoder outputs
        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)
        
        # Encode source sequences
        encoder_outputs, hidden, cell = self.encoder(src, src_len)
        
        # First input to the decoder is the <SOS> token
        input = tgt[:, 0].unsqueeze(1) # [batch_size, 1]
        
        for t in range(1, tgt_len):
            prediction, hidden, cell, _ = self.decoder(input, hidden, cell, encoder_outputs)
            
            # Save prediction
            outputs[:, t, :] = prediction
            
            # Decide if we are doing teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            
            # Get the highest predicted token from our predictions
            top1 = prediction.argmax(1) 
            
            # If teacher forcing, use actual next token as next input
            # If not, use predicted token
            input = tgt[:, t].unsqueeze(1) if teacher_force else top1.unsqueeze(1)
            
        return outputs
