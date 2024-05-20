import torch
import json
from mamba import Mamba

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
def load_tokenizer(filepath):
    with open(filepath, 'r') as file:
        tokenizer = json.load(file)
    return tokenizer

tokenizer_en = load_tokenizer('/mnt/data/tokenizer_en.json')
tokenizer_it = load_tokenizer('/mnt/data/tokenizer_it.json')

# Tokenizer utility functions
def tokenize(sentence, tokenizer):
    tokens = [tokenizer['model']['vocab'].get(word, tokenizer['model']['vocab']['[UNK]']) for word in sentence.split()]
    tokens = [tokenizer['model']['vocab']['[SOS]']] + tokens + [tokenizer['model']['vocab']['[EOS]']]
    return torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(DEVICE)  # Add batch dimension

def detokenize(tokens, tokenizer):
    vocab = {v: k for k, v in tokenizer['model']['vocab'].items()}
    words = [vocab[token.item()] for token in tokens if token.item() in vocab]
    return ' '.join(words).replace('[SOS]', '').replace('[EOS]', '').strip()

# Initialize the model
model = torch.load_state_dict(torch.load('model_weights.pth'))

# Dummy input for model testing (Replace with actual sentence input)
sentence = "This is a test sentence."

# Tokenize input
input_tokens = tokenize(sentence, tokenizer_en)

# Run the model on the input tokens
with torch.no_grad():
    output_tokens = model(input_tokens)

# Get the most probable tokens (argmax)
output_tokens = torch.argmax(output_tokens, dim=-1).squeeze()

# Detokenize output
translated_sentence = detokenize(output_tokens, tokenizer_it)

# Print the output
print("Input Sentence:", sentence)
print("Translated Sentence:", translated_sentence)