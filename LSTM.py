import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from nltk.tokenize import word_tokenize
from collections import Counter
import numpy as np
import nltk
import time
import matplotlib.pyplot as plt

# Download punkt untuk tokenization
nltk.download('punkt')

# Load data training
train_data = pd.read_csv(
    r"C:\Users\HP 840 G8\Downloads\ag_news_csv\ag_news_csv\data\train.csv",
    header=None,
    names=["label", "title", "content"]
)

# Load data testing
test_data = pd.read_csv(
    r"C:\Users\HP 840 G8\Downloads\ag_news_csv\ag_news_csv\data\test.csv",
    header=None,
    names=["label", "title", "content"]
)

# Kombinasi judul dan konten ke dalam single kolom
train_data["text"] = train_data["title"] + " " + train_data["content"]
test_data["text"] = test_data["title"] + " " + test_data["content"]

# Atur labels dimulai dari 0
train_data["label"] = train_data["label"] - 1
test_data["label"] = test_data["label"] - 1

# Bangun vocabulary
def build_vocab(texts):
    tokenized_texts = [word_tokenize(text.lower()) for text in texts]
    word_counts = Counter([word for tokens in tokenized_texts for word in tokens])
    vocab = {word: idx + 1 for idx, (word, _) in enumerate(word_counts.most_common())}
    vocab["<PAD>"] = 0
    return vocab

# Load pre-trained embeddings
def load_pretrained_embeddings(vocab, embedding_path, embedding_dim):
    embeddings = np.zeros((len(vocab), embedding_dim))
    with open(embedding_path, "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype="float32")
            if word in vocab:
                idx = vocab[word]
                embeddings[idx] = vector
    return torch.tensor(embeddings, dtype=torch.float)

# Tokenisasi dan bangun vocabulary
vocab = build_vocab(train_data["text"])

# Convert teks ke indeks
def text_to_indices(text, vocab, max_len=100):
    tokens = word_tokenize(text.lower())
    indices = [vocab.get(word, 0) for word in tokens]
    return indices[:max_len] + [0] * (max_len - len(indices))

train_data["text_indices"] = train_data["text"].apply(lambda x: text_to_indices(x, vocab))
test_data["text_indices"] = test_data["text"].apply(lambda x: text_to_indices(x, vocab))

# GloVe
embedding_path = r"C:\Users\HP 840 G8\Downloads\ag_news_csv\ag_news_csv\data\glove.6B.100d.txt"
embedding_dim = 100
pretrained_embeddings = load_pretrained_embeddings(vocab, embedding_path, embedding_dim)

# Custom Dataset untuk DataLoader
class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return torch.tensor(self.texts.iloc[idx], dtype=torch.long), torch.tensor(self.labels.iloc[idx], dtype=torch.long)

# DataLoader
batch_size = 32
train_dataset = TextDataset(train_data["text_indices"], train_data["label"])
test_dataset = TextDataset(test_data["text_indices"], test_data["label"])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Definisi Model BiLSTM
class BiLSTM(nn.Module):
    def __init__(self, n_classes, vocab_size, embeddings, emb_size, fine_tune, rnn_size, rnn_layers, dropout):
        super(BiLSTM, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, emb_size)
        self.set_embeddings(embeddings, fine_tune)
        self.BiLSTM = nn.LSTM(emb_size, rnn_size, num_layers=rnn_layers, bidirectional=True,
                              dropout=(0 if rnn_layers == 1 else dropout), batch_first=True)
        self.fc = nn.Linear(2 * rnn_size, n_classes)
        self.dropout = nn.Dropout(dropout)

    def set_embeddings(self, embeddings, fine_tune=True):
        if embeddings is None:
            self.embeddings.weight.data.uniform_(-0.1, 0.1)
        else:
            self.embeddings.weight = nn.Parameter(embeddings, requires_grad=fine_tune)

    def forward(self, text, words_per_sentence):
        embeddings = self.dropout(self.embeddings(text))
        packed_words = pack_padded_sequence(embeddings, lengths=words_per_sentence.tolist(),
                                            batch_first=True, enforce_sorted=False)
        rnn_out, _ = self.BiLSTM(packed_words)
        rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True)
        H = torch.mean(rnn_out, dim=1)
        scores = self.fc(self.dropout(H))
        return scores

# Inisialisasi model
model = BiLSTM(
    n_classes=len(train_data["label"].unique()),
    vocab_size=len(vocab),
    embeddings=pretrained_embeddings,
    emb_size=embedding_dim,
    fine_tune=True,
    rnn_size=128,
    rnn_layers=2,
    dropout=0.5
)

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Inisialisasi list untuk menyimpan hasil per epoch
train_losses = []
train_accuracies = []
epoch_times = []

# Training loop
for epoch in range(5):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    start_time = time.time()

    for texts, labels in train_loader:
        optimizer.zero_grad()
        words_per_sentence = (texts != 0).sum(dim=1)
        outputs = model(texts, words_per_sentence)

        # Hitung loss
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

        # Hitung akurasi
        _, predicted = outputs.max(1)
        correct_predictions += (predicted == labels).sum().item()
        total_samples += labels.size(0)

    elapsed_time = time.time() - start_time
    accuracy = (correct_predictions / total_samples) * 100
    avg_loss = total_loss / len(train_loader)

    epoch_times.append(elapsed_time)
    train_losses.append(avg_loss)
    train_accuracies.append(accuracy)

    print(f"Epoch {epoch + 1}, Rata-rata Loss: {avg_loss:.4f}, Akurasi: {accuracy:.2f}%, Waktu: {elapsed_time:.2f} detik")
