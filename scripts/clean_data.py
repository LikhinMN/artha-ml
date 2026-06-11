import json
import os
import unicodedata

def clean_data(input_path, output_path):
    print(f"Loading data from {input_path}...")
    
    # Try parsing as standard JSON, fallback to JSON Lines
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        data = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    
    print(f"Original pairs: {len(data)}")
    
    cleaned_data = []
    seen_pairs = set()
    
    for item in data:
        en = item.get('en', '')
        kn = item.get('kn', '')
        
        # 1. Remove pairs where either side is empty
        if not en or not kn:
            continue
            
        # 5. Normalize unicode (NFC for Indic languages)
        en = unicodedata.normalize('NFC', str(en).strip())
        kn = unicodedata.normalize('NFC', str(kn).strip())
        
        # Check again if empty after stripping
        if not en or not kn:
            continue
            
        # 3. Remove duplicate pairs
        pair_tuple = (en, kn)
        if pair_tuple in seen_pairs:
            continue
            
        # Word counts
        en_words = en.split()
        kn_words = kn.split()
        en_len = len(en_words)
        kn_len = len(kn_words)
        
        # 2. Filter sentences shorter than 3 words or longer than 50 words
        if not (3 <= en_len <= 50) or not (3 <= kn_len <= 50):
            continue
            
        # 4. Check length ratio (en_len / kn_len should be between 0.3 and 3.0)
        # Prevent division by zero (though handled by previous conditions)
        if kn_len == 0:
            continue
            
        ratio = en_len / kn_len
        if not (0.3 <= ratio <= 3.0):
            continue
            
        seen_pairs.add(pair_tuple)
        cleaned_data.append({'en': en, 'kn': kn})
        
    final_count = len(cleaned_data)
    removed_count = len(data) - final_count
    
    print(f"Original count: {len(data)}")
    print(f"Removed count: {removed_count}")
    print(f"Final count: {final_count}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Saving cleaned data to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
    print("Done!")

if __name__ == "__main__":
    input_file = "data/raw/samanantar_kn_50k.json"
    output_file = "data/processed/clean_50k.json"
    
    # Ensure working from project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    in_path = os.path.join(project_root, input_file)
    out_path = os.path.join(project_root, output_file)
    
    if not os.path.exists(in_path):
        print(f"Error: Input file not found at {in_path}")
        print("Please ensure the data is downloaded and placed correctly.")
    else:
        clean_data(in_path, out_path)
