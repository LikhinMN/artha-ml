import os
import torch
from torch.utils.data import DataLoader

# Add project root to python path to import arthaml
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from arthaml.data.vocabulary import Vocabulary
from arthaml.data.dataset import TranslationDataset

def main():
    print("Loading vocabularies...")
    vocab_en = Vocabulary("English")
    vocab_en.load(os.path.join(project_root, 'data/processed/vocab_en.json'))
    print(f"Loaded English vocab size: {len(vocab_en.word2idx)}")
    
    vocab_kn = Vocabulary("Kannada")
    vocab_kn.load(os.path.join(project_root, 'data/processed/vocab_kn.json'))
    print(f"Loaded Kannada vocab size: {len(vocab_kn.word2idx)}")
    
    print("Creating dataset and dataloader...")
    data_path = os.path.join(project_root, 'data/processed/clean_50k.json')
    dataset = TranslationDataset(data_path, vocab_en, vocab_kn, max_len=50)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    print(f"\nTotal samples: {len(dataset)}")
    print(f"Total batches: {len(dataloader)}")
    
    print("\nVerifying first 3 batches:")
    for i, batch in enumerate(dataloader):
        if i >= 3:
            break
        print(f"Batch {i+1}:")
        print(f"  src shape: {batch['src'].shape}")
        print(f"  tgt shape: {batch['tgt'].shape}")
        
        if i == 0:
            print("\nDecoded sample from Batch 1:")
            src_indices = batch['src'][0].tolist()
            tgt_indices = batch['tgt'][0].tolist()
            print(f"  src length: {batch['src_len'][0]}")
            print(f"  src decoded: {vocab_en.decode(src_indices)}")
            print(f"  tgt length: {batch['tgt_len'][0]}")
            print(f"  tgt decoded: {vocab_kn.decode(tgt_indices)}")
            print()

if __name__ == "__main__":
    main()
