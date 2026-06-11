import torch
import sacrebleu
import random

def greedy_decode(model, src, src_len, vocab_kn, device, max_len=50):
    model.eval()
    with torch.no_grad():
        # src: [1, src_len]
        encoder_outputs, hidden, cell = model.encoder(src, src_len)
        
        input = torch.tensor([[vocab_kn.word2idx['<SOS>']]], device=device)
        
        decoded_words = []
        
        for t in range(max_len):
            prediction, hidden, cell, _ = model.decoder(input, hidden, cell, encoder_outputs)
            
            top1 = prediction.argmax(1)
            token = top1.item()
            
            if token == vocab_kn.word2idx['<EOS>']:
                break
                
            if token not in [vocab_kn.word2idx['<SOS>'], vocab_kn.word2idx['<PAD>']]:
                decoded_words.append(vocab_kn.idx2word.get(token, '<UNK>'))
                
            input = top1.unsqueeze(1)
            
        return " ".join(decoded_words)

def compute_bleu(model, dataset, vocab_kn, device, n=500):
    model.eval()
    
    # Sample n items from the dataset
    num_samples = min(n, len(dataset))
    indices = random.sample(range(len(dataset)), num_samples)
    
    references = []
    predictions = []
    
    for idx in indices:
        item = dataset[idx]
        src = item['src'].unsqueeze(0).to(device)
        src_len = torch.tensor([item['src_len']]).to(device)
        
        tgt_indices = item['tgt'].tolist()
        tgt_words = []
        for token in tgt_indices:
            if token == vocab_kn.word2idx['<EOS>']:
                break
            if token not in [vocab_kn.word2idx['<SOS>'], vocab_kn.word2idx['<PAD>']]:
                tgt_words.append(vocab_kn.idx2word.get(token, '<UNK>'))
                
        references.append(" ".join(tgt_words))
        
        pred = greedy_decode(model, src, src_len, vocab_kn, device)
        predictions.append(pred)
        
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    return bleu.score
