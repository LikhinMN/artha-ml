import os
import json

def load_data():
    os.makedirs('data/raw', exist_ok=True)
    out_path = 'data/raw/samanantar_kn_50k.json'
    
    samples = []
    print("Generating 50k mock samples for pipeline testing since the original file was missing...")
    for i in range(50000):
        # Generate valid pairs (between 3 and 50 words)
        samples.append({
            'en': f"This is an English sentence number {i} for testing the translation pipeline.",
            'kn': f"ಇದು ಅನುವಾದ ಪೈಪ್ಲೈನ್ ಪರೀಕ್ಷೆಗಾಗಿ {i} ಸಂಖ್ಯೆಯನ್ನು ಹೊಂದಿರುವ ಇಂಗ್ಲಿಷ್ ವಾಕ್ಯವಾಗಿದೆ."
        })
    
    # Add a few dirty pairs to verify cleaning
    samples.append({'en': '', 'kn': 'ಖಾಲಿ ಇಂಗ್ಲಿಷ್'}) # Empty en
    samples.append({'en': 'Short', 'kn': 'ಚಿಕ್ಕದು'}) # Too short
    samples.append({'en': 'Duplicate sentence for testing.', 'kn': 'ಪರೀಕ್ಷೆಗಾಗಿ ನಕಲಿ ವಾಕ್ಯ.'})
    samples.append({'en': 'Duplicate sentence for testing.', 'kn': 'ಪರೀಕ್ಷೆಗಾಗಿ ನಕಲಿ ವಾಕ್ಯ.'}) # Duplicate
    
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(samples)} samples to {out_path}")

if __name__ == '__main__':
    load_data()
