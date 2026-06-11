import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_size=512):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)
        
    def forward(self, hidden, encoder_outputs):
        # hidden: [num_layers, batch_size, hidden_size]
        # encoder_outputs: [batch_size, seq_len, hidden_size]
        
        batch_size = encoder_outputs.shape[0]
        seq_len = encoder_outputs.shape[1]
        
        # We use the final layer's hidden state for attention
        hidden_last = hidden[-1] # [batch_size, hidden_size]
        
        # Repeat hidden state across seq_len
        # [batch_size, seq_len, hidden_size]
        hidden_repeated = hidden_last.unsqueeze(1).repeat(1, seq_len, 1)
        
        # energy: [batch_size, seq_len, hidden_size * 2]
        energy = torch.cat((hidden_repeated, encoder_outputs), dim=2)
        
        # attention energy: [batch_size, seq_len, hidden_size]
        energy = torch.tanh(self.attn(energy))
        
        # attention weights: [batch_size, seq_len]
        attention = self.v(energy).squeeze(2)
        
        return F.softmax(attention, dim=1)

class Decoder(nn.Module):
    def __init__(self, attention, vocab_size=26802, embed_dim=256, hidden_size=512, num_layers=2, dropout_p=0.3, padding_idx=0):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.attention = attention
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.dropout = nn.Dropout(dropout_p)
        
        # LSTM input size = embed_dim + context (hidden_size) = 768
        self.lstm = nn.LSTM(embed_dim + hidden_size, hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout_p if num_layers > 1 else 0)
        
        # fc_out = hidden + context + embed = 512 + 512 + 256 = 1280 -> vocab_size
        self.fc_out = nn.Linear(hidden_size * 2 + embed_dim, vocab_size)
        
    def forward(self, input, hidden, cell, encoder_outputs):
        # input: [batch_size, 1]
        # hidden, cell: [num_layers, batch_size, hidden_size]
        # encoder_outputs: [batch_size, seq_len, hidden_size]
        
        embedded = self.dropout(self.embedding(input)) # [batch_size, 1, embed_dim]
        
        # attention weights: [batch_size, seq_len]
        a = self.attention(hidden, encoder_outputs).unsqueeze(1) # [batch_size, 1, seq_len]
        
        # context: [batch_size, 1, hidden_size]
        context = torch.bmm(a, encoder_outputs)
        
        # concat embedded input and context
        lstm_input = torch.cat((embedded, context), dim=2) # [batch_size, 1, embed_dim + hidden_size]
        
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell)) # output: [batch_size, 1, hidden_size]
        
        # fc_input: [batch_size, hidden_size * 2 + embed_dim]
        fc_input = torch.cat((output, context, embedded), dim=2).squeeze(1)
        
        prediction = self.fc_out(fc_input) # [batch_size, vocab_size]
        
        return prediction, hidden, cell, a.squeeze(1)
