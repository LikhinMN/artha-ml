import os
import json
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from arthaml.data.vocabulary import Vocabulary

def main():
    input_file = os.path.join(project_root, "data/processed/clean_50k.json")
    
    if not os.path.exists(input_file):
        print(f"Error: Cleaned data not found at {input_file}")
        return
        
    print("Loading cleaned data...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    en_sentences = [item['en'] for item in data]
    kn_sentences = [item['kn'] for item in data]
    
    print("Building English vocabulary...")
    vocab_en = Vocabulary("English", min_freq=2)
    vocab_en.build(en_sentences)
    vocab_en.save(os.path.join(project_root, 'data/processed/vocab_en.json'))
    
    print("Building Kannada vocabulary...")
    vocab_kn = Vocabulary("Kannada", min_freq=2)
    vocab_kn.build(kn_sentences)
    vocab_kn.save(os.path.join(project_root, 'data/processed/vocab_kn.json'))
    
    print(f"\nFinal vocab sizes:")
    print(f"English: {len(vocab_en.word2idx)}")
    print(f"Kannada: {len(vocab_kn.word2idx)}")

if __name__ == "__main__":
    main()
