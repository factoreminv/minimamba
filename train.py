import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
import torch.nn.functional as F

from mamba import Mamba

from time import time

# Load tokenizer function
def load_tokenizer(path):
    with open(path, 'r', encoding="utf-8") as file:
        data = json.load(file)
    vocab = data['model']['vocab']
    return vocab

# Tokenization function
def tokenize(text, vocab):
    tokens = text.lower().split()  # Simple whitespace tokenizer
    return [vocab.get(token, vocab.get('[UNK]')) for token in tokens]

# Dataset class
class TranslationDataset(Dataset):
    def __init__(self, dataset, tokenizer_en, tokenizer_it):
        self.dataset = dataset
        self.tokenizer_en = tokenizer_en
        self.tokenizer_it = tokenizer_it

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        src_text = self.dataset[idx]['en']
        trg_text = self.dataset[idx]['it']
        src_encoded = tokenize(src_text, self.tokenizer_en)
        trg_encoded = tokenize(trg_text, self.tokenizer_it)
        return {'src': torch.tensor(src_encoded, dtype=torch.long),
                'trg': torch.tensor(trg_encoded, dtype=torch.long)}

def collate_fn(batch):
    src_batch = pad_sequence([item['src'] for item in batch], batch_first=True, padding_value=token_to_id_en['[PAD]'])
    trg_batch = pad_sequence([item['trg'] for item in batch], batch_first=True, padding_value=token_to_id_it['[PAD]'])

    max_seq_len = max(len(src_batch[0]), len(trg_batch[0]))

    src_batch = F.pad(src_batch, (0,max_seq_len-len(src_batch[0]),0,0), value=token_to_id_en['[PAD]'])
    trg_batch = F.pad(trg_batch, (0,max_seq_len-len(trg_batch[0]),0,0), value=token_to_id_en['[PAD]'])

    print(len(src_batch[0]), len(trg_batch[0]))
    return src_batch.to(device), trg_batch.to(device)

# Load English and Italian tokenizers
tokenizer_en_path = 'tokenizer_en.json'
tokenizer_it_path = 'tokenizer_it.json'
token_to_id_en = load_tokenizer(tokenizer_en_path)
token_to_id_it = load_tokenizer(tokenizer_it_path)

# Load dataset from Hugging Face datasets
dataset = load_dataset("opus_books", "en-it")
dataset = dataset.map(lambda examples: {'en': examples['translation']['en'], 'it': examples['translation']['it']})

# Create the translation dataset
train_dataset = TranslationDataset(dataset['train'], token_to_id_en, token_to_id_it)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

device = torch.device("cpu" if torch.backends.mps.is_available() else "cpu")

# Model definition (assuming Mamba and its sub-components are defined elsewhere in your setup)
model = Mamba(512, 6, len(token_to_id_it), 128, 3, 2, 10)
model.to(device)
# Loss and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=token_to_id_it['[PAD]'])
optimizer = optim.Adam(model.parameters())

# Training loop
def train(model, data_loader, optimizer, criterion, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for src, trg in data_loader:
            optimizer.zero_grad()
            output = model(src)  # output shape might be [batch_size, seq_len, vocab_size]
            output_flat = output.reshape(-1, output.shape[-1])
            trg_flat = trg.reshape(-1)
            loss = criterion(output_flat, trg_flat)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch}, Loss: {total_loss / len(data_loader)}")


# Train the model
num_epochs = 10
train(model, train_loader, optimizer, criterion, num_epochs)
