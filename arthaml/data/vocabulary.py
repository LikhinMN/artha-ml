import os
import json
from collections import Counter

class Vocabulary:
    def __init__(self, name: str, min_freq: int = 2):
        self.name = name
        self.min_freq = min_freq
        self.pad_token = '<PAD>'
        self.sos_token = '<SOS>'
        self.eos_token = '<EOS>'
        self.unk_token = '<UNK>'
        
        # Initialize token mappings with special tokens
        self.word2idx = {
            self.pad_token: 0,
            self.sos_token: 1,
            self.eos_token: 2,
            self.unk_token: 3
        }
        self.idx2word = {
            0: self.pad_token,
            1: self.sos_token,
            2: self.eos_token,
            3: self.unk_token
        }
        self.word_freqs = Counter()

    def build(self, sentences: list[str]):
        """Build vocabulary from a list of sentences (word-level tokenization)."""
        for sentence in sentences:
            self.word_freqs.update(sentence.split())
            
        for word, freq in self.word_freqs.items():
            if freq >= self.min_freq:
                if word not in self.word2idx:
                    idx = len(self.word2idx)
                    self.word2idx[word] = idx
                    self.idx2word[idx] = word

    def encode(self, sentence: str) -> list[int]:
        """Convert a string sentence into a list of vocabulary indices."""
        return [self.word2idx.get(word, self.word2idx[self.unk_token]) for word in sentence.split()]

    def decode(self, indices: list[int]) -> str:
        """Convert a list of indices back to a string sentence."""
        return " ".join([self.idx2word.get(idx, self.unk_token) for idx in indices])

    def save(self, path: str):
        """Save the vocabulary to a JSON file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'word2idx': self.word2idx,
                'idx2word': self.idx2word,
                'word_freqs': dict(self.word_freqs)
            }, f, ensure_ascii=False, indent=2)
        print(f"[{self.name}] Vocabulary saved to {path}. Size: {len(self.word2idx)}")

    def load(self, path: str):
        """Load the vocabulary from a JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.word2idx = data['word2idx']
        self.idx2word = {int(k): v for k, v in data['idx2word'].items()}
        self.word_freqs = Counter(data.get('word_freqs', {}))
        print(f"[{self.name}] Vocabulary loaded from {path}. Size: {len(self.word2idx)}")

if __name__ == "__main__":
    # Ensure working from project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    input_file = os.path.join(project_root, "data/processed/clean_50k.json")
    
    if not os.path.exists(input_file):
        print(f"Error: Cleaned data not found at {input_file}")
    else:
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
