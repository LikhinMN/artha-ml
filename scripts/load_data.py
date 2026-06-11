import os
import json
from datasets import load_dataset

def load_data():
    os.makedirs('data/raw', exist_ok=True)
    out_path = 'data/raw/samanantar_kn_50k.json'

    print("Downloading ai4bharat/samanantar kn subset...")
    ds = load_dataset('ai4bharat/samanantar', 'kn', split='train', streaming=True)
    
    samples = []
    print("Fetching 50k samples...")
    for i, item in enumerate(ds):
        if i >= 50000:
            break
        en = item.get('src', item.get('en', item.get('english', '')))
        kn = item.get('tgt', item.get('kn', item.get('kannada', '')))
        samples.append({'en': en, 'kn': kn})
    
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    print(f"Saved 50k samples to {out_path}")

if __name__ == '__main__':
    load_data()
