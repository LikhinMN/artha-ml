import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Encoder(nn.Module):
    def __init__(self, vocab_size=21551, embed_dim=256, hidden_size=512, num_layers=2, dropout_p=0.3, padding_idx=0):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.dropout = nn.Dropout(dropout_p)
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout_p if num_layers > 1 else 0)
        
    def forward(self, src, src_len):
        # src: [batch_size, seq_len]
        # src_len: [batch_size]
        
        embedded = self.dropout(self.embedding(src))
        
        # Move lengths to CPU for pack_padded_sequence as required by PyTorch
        src_len_cpu = src_len.cpu()
        
        # Pack sequence for efficiency
        packed_embedded = pack_padded_sequence(embedded, src_len_cpu, batch_first=True, enforce_sorted=False)
        
        packed_outputs, (hidden, cell) = self.lstm(packed_embedded)
        
        # Unpack sequence
        outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True)
        # outputs: [batch_size, seq_len, hidden_size]
        # hidden, cell: [num_layers, batch_size, hidden_size]
        
        return outputs, hidden, cell
