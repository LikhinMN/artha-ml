import json
import torch
from torch.utils.data import Dataset
from arthaml.data.vocabulary import Vocabulary

class TranslationDataset(Dataset):
    def __init__(self, data_path: str, vocab_en: Vocabulary, vocab_kn: Vocabulary, max_len: int = 50):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.vocab_en = vocab_en
        self.vocab_kn = vocab_kn
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Encode src (English) and tgt (Kannada)
        src_encoded = self.vocab_en.encode(item['en'])
        tgt_encoded = self.vocab_kn.encode(item['kn'])
        
        # Truncate if too long, leaving room for <SOS> and <EOS>
        if len(src_encoded) > self.max_len - 2:
            src_encoded = src_encoded[:self.max_len - 2]
        if len(tgt_encoded) > self.max_len - 2:
            tgt_encoded = tgt_encoded[:self.max_len - 2]
            
        # Prepend <SOS> and append <EOS>
        src_full = [self.vocab_en.word2idx['<SOS>']] + src_encoded + [self.vocab_en.word2idx['<EOS>']]
        tgt_full = [self.vocab_kn.word2idx['<SOS>']] + tgt_encoded + [self.vocab_kn.word2idx['<EOS>']]
        
        src_len = len(src_full)
        tgt_len = len(tgt_full)
        
        # Pad to exactly max_len
        src_padded = src_full + [self.vocab_en.word2idx['<PAD>']] * (self.max_len - src_len)
        tgt_padded = tgt_full + [self.vocab_kn.word2idx['<PAD>']] * (self.max_len - tgt_len)
        
        return {
            "src": torch.tensor(src_padded, dtype=torch.long),
            "tgt": torch.tensor(tgt_padded, dtype=torch.long),
            "src_len": src_len,
            "tgt_len": tgt_len
        }
