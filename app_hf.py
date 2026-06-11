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
    
    # If translation contains more than 3 consecutive <UNK> tokens (which means 4 or more),
    # or exactly 4 <UNK> tokens consecutively, we flag it.
    if "<UNK> <UNK> <UNK> <UNK>" in translation:
        translation = "⚠️ Translation confidence is low — try simpler or shorter sentences."
        
    elapsed = (time.time() - start) * 1000
    return translation, f"{elapsed:.2f} ms"

def check_length(text):
    if len(text.split()) > 20:
        return "💡 Tip: This model works best with sentences under 20 words."
    return ""

with gr.Blocks() as demo:
    gr.Markdown("# ArthaML — Kannada-English Translator")
    gr.Markdown("Open-source neural machine translation built from scratch. Best results with simple, everyday English sentences under 20 words.")
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(lines=3, placeholder="Enter English text", label="English text")
            warning_msg = gr.Markdown("")
        with gr.Column():
            output_text = gr.Textbox(label="Kannada translation")
            time_text = gr.Textbox(label="Time taken")
            
    input_text.change(fn=check_length, inputs=input_text, outputs=warning_msg)
    
    translate_btn = gr.Button("Translate")
    translate_btn.click(fn=translate_interface, inputs=input_text, outputs=[output_text, time_text])
    
    gr.Examples(
        examples=[
            ["Hello, how are you?"],
            ["The weather is beautiful today."],
            ["Education is the key to success."],
            ["I love my family."],
            ["India is a great country."],
            ["The train arrives at six o'clock."]
        ],
        inputs=input_text
    )

if __name__ == "__main__":
    demo.launch(share=False)
