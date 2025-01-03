!pip install gensim

!pip install gensim torch torchvision scikit-learn pandas

# Import Library
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import nn
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import os

# Cek device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Path Dataset dan GloVe
TRAIN_PATH = "train.csv"
TEST_PATH = "test.csv"
GLOVE_PATH = "/content/glove.6B.100d.txt"  # Path ke file GloVe
EMBEDDING_DIM = 100

# Load Dataset
train_df = pd.read_csv(TRAIN_PATH, header=None, names=["class", "title", "description"])
test_df = pd.read_csv(TEST_PATH, header=None, names=["class", "title", "description"])

# Gabungkan title dan description
train_df["text"] = train_df["title"].fillna("") + ". " + train_df["description"].fillna("")
test_df["text"] = test_df["title"].fillna("") + ". " + test_df["description"].fillna("")

# Encode label
label_encoder = LabelEncoder()
train_df["label"] = label_encoder.fit_transform(train_df["class"])
test_df["label"] = label_encoder.transform(test_df["class"])

print("Sample data:")
print(train_df.head())

# Load GloVe embeddings
def load_glove_embeddings(glove_path, embedding_dim):
    word_to_idx = {"<PAD>": 0, "<UNK>": 1}
    embedding_matrix = [np.zeros(embedding_dim), np.random.uniform(-0.25, 0.25, embedding_dim)]

    print("Loading GloVe embeddings...")
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            word = parts[0]
            vector = np.array(parts[1:], dtype=np.float32)
            word_to_idx[word] = len(word_to_idx)
            embedding_matrix.append(vector)

    embedding_matrix = np.array(embedding_matrix)
    print("GloVe embeddings loaded successfully!")
    return word_to_idx, torch.tensor(embedding_matrix, dtype=torch.float)

word_to_idx, embedding_matrix = load_glove_embeddings(GLOVE_PATH, EMBEDDING_DIM)
print(f"Embedding matrix shape: {embedding_matrix.shape}")

# Tokenizer sederhana
def tokenize(text):
    return text.lower().split()

# Dataset class
class AGNewsDataset(Dataset):
    def __init__(self, texts, labels, word_to_idx, max_len):
        self.texts = texts
        self.labels = labels
        self.word_to_idx = word_to_idx
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # Tokenisasi dan konversi ke indeks
        tokens = tokenize(text)
        token_ids = [self.word_to_idx.get(token, 1) for token in tokens]  # 1 = "<UNK>"

        # Padding
        if len(token_ids) < self.max_len:
            token_ids += [0] * (self.max_len - len(token_ids))  # 0 = "<PAD>"
        else:
            token_ids = token_ids[:self.max_len]

        return {
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long)
        }

# Dataset dan DataLoader
MAX_LEN = 100
train_dataset = AGNewsDataset(train_df["text"].values, train_df["label"].values, word_to_idx, MAX_LEN)
test_dataset = AGNewsDataset(test_df["text"].values, test_df["label"].values, word_to_idx, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

class TransformerClassifier(nn.Module):
    def __init__(self, embedding_matrix, n_heads, hidden_size, n_encoders, n_classes, dropout=0.5):
        super(TransformerClassifier, self).__init__()
        vocab_size, embedding_dim = embedding_matrix.shape

        # Load GloVe embeddings
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=True)
        self.positional_encoding = nn.Embedding(MAX_LEN, embedding_dim)

        # Transformer Encoder
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=n_heads, dim_feedforward=hidden_size, dropout=dropout)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=n_encoders)

        self.fc = nn.Linear(embedding_dim * MAX_LEN, n_classes)

    def forward(self, x):
        x = self.embedding(x) + self.positional_encoding(torch.arange(MAX_LEN, device=DEVICE))
        x = self.encoder(x.permute(1, 0, 2))  # (seq_len, batch_size, emb_dim)
        x = x.permute(1, 0, 2).contiguous().view(x.size(1), -1)  # Flatten
        return self.fc(x)

# Inisialisasi Model
model = TransformerClassifier(
    embedding_matrix=embedding_matrix,
    n_heads=4,
    hidden_size=256,
    n_encoders=2,
    n_classes=len(label_encoder.classes_)
).to(DEVICE)

# Optimizer dan Loss
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# Training Function
def train_epoch(model, data_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for batch in data_loader:
        input_ids = batch["input_ids"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        optimizer.zero_grad()
        outputs = model(input_ids)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)

# Evaluation function
def evaluate(model, data_loader):
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            outputs = model(input_ids)
            preds = torch.argmax(outputs, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    return accuracy_score(true_labels, predictions)

# Train model with timing
import time

epochs = 5
for epoch in range(epochs):
    start_time = time.time()
    train_loss = train_epoch(model, train_loader, optimizer, criterion)
    acc = evaluate(model, test_loader)
    end_time = time.time()
    epoch_time = end_time - start_time
    print(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Test Accuracy: {acc:.4f}, Time: {epoch_time:.2f}s")

# Save model
torch.save(model.state_dict(), "transformer_with_glove.pth")
print("Model saved successfully!")
