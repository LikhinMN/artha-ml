import gradio as gr
import os
import torch
import time
from arthaml.inference import Translator

project_root = os.path.abspath(os.path.dirname(__file__))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = os.path.join(project_root, 'models/best_model.pth')
vocab_en_path = os.path.join(project_root, 'data/processed/vocab_en.json')
vocab_kn_path = os.path.join(project_root, 'data/processed/vocab_kn.json')

translator = Translator(model_path, vocab_en_path, vocab_kn_path, device)

def translate_interface(text):
    start = time.time()
    translation = translator.translate(text)
    
    if "<UNK> <UNK> <UNK> <UNK>" in translation:
        translation = "⚠️ Translation confidence is low — try simpler or shorter sentences."
        
    elapsed = (time.time() - start) * 1000
    return translation, f"{elapsed:.2f} ms"

demo = gr.Interface(
    fn=translate_interface,
    inputs=gr.Textbox(lines=3, placeholder="Enter English text", label="English text"),
    outputs=[
        gr.Textbox(label="Kannada translation"),
        gr.Textbox(label="Time taken")
    ],
    title="ArthaML — Kannada-English Translator",
    description="Open-source neural machine translation built from scratch. Best results with simple, everyday English sentences under 20 words.",
    examples=[
        ["Hello, how are you?"],
        ["The weather is beautiful today."],
        ["Education is the key to success."],
        ["I love my family."],
        ["India is a great country."],
        ["The train arrives at six o'clock."]
    ]
)

if __name__ == "__main__":
    demo.launch(share=False)
