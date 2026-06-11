import os
import sys
import argparse
import time
import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from arthaml.inference import Translator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('text', nargs='?', type=str, help='English text to translate')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_path = os.path.join(project_root, 'models/best_model.pth')
    vocab_en_path = os.path.join(project_root, 'data/processed/vocab_en.json')
    vocab_kn_path = os.path.join(project_root, 'data/processed/vocab_kn.json')
    
    if not os.path.exists(model_path):
        print(f"Error: Could not find model at {model_path}")
        return
        
    print("Loading Translator...")
    start_load = time.time()
    translator = Translator(model_path, vocab_en_path, vocab_kn_path, device)
    print(f"Loaded in {(time.time() - start_load)*1000:.0f}ms")
    
    if args.interactive:
        print("Interactive mode started. Enter 'q' to quit.")
        while True:
            try:
                text = input("Enter English text (q to quit): ")
                if text.lower().strip() == 'q':
                    break
                
                start = time.time()
                translation = translator.translate(text)
                elapsed = (time.time() - start) * 1000
                print(f"Kannada translation: {translation}")
                print(f"Time taken: {elapsed:.2f} ms")
            except (KeyboardInterrupt, EOFError):
                break
    elif args.text:
        start = time.time()
        translation = translator.translate(args.text)
        elapsed = (time.time() - start) * 1000
        print(f"Kannada translation: {translation}")
        print(f"Time taken: {elapsed:.2f} ms")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
