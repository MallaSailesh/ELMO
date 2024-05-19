import torch 
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from collections import Counter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ELMO(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, batch_size, words_to_ind):

        super(ELMO, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.words_to_ind = words_to_ind
        '''
        By creating an instance of nn.Embedding with the parameters vocab_size and embedding_dim , the 
        self.embedding becomes a trainable layer in neural network , initialized with random values. During training ,
        the values of the embedding vectors will be adjusted through backpropagation to minimize the loss function. 
        '''
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)

        # 2 layers of bilstm 
        # bidirectional=True => forward and backward  ,  batch_first = True => it expects the first dimension as batch_size 
        self.lstm1 = nn.LSTM(embedding_dim, hidden_dim,batch_first=True, bidirectional=True) 
        '''
        Here hidden_dim is the size produced by both forward and backward pass which when concatenated gives hidden_dim*2. 
        This is because we have both the forward and backward hidden states for each time step concatenated together.
        '''
        self.lstm2 = nn.LSTM(hidden_dim*2, hidden_dim,batch_first=True, bidirectional=True)
        self.linear_out = nn.Linear(hidden_dim*2, vocab_size)

    def forward(self, x):
        x_embed = self.embedding(x)
        x_lstm1, _ = self.lstm1(x_embed)
        x_lstm2, _ = self.lstm2(x_lstm1)
        linear_out = self.linear_out(x_lstm2)
        return linear_out
    
    def train(self, train_tokens_indices, num_epochs=5, lr=0.001):

        max_length = max(len(seq) for seq in train_tokens_indices) # Determine the maximum length of sequences
        padded_sequences = [torch.tensor(seq + [0] * (max_length - len(seq)), dtype=torch.long) for seq in train_tokens_indices] # Pad sequences to the maximum length
        train_tokens_tensor = torch.stack(padded_sequences) # Stack padded sequences into a tensor
        train_dataset = TensorDataset(train_tokens_tensor) # Create a Tensor dataset
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True) # Now Create a data loader

        self.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)

        for epoch in range(num_epochs):
    
            running_loss = 0.0

            for batch in tqdm(train_dataloader, desc="Epoch {} / {}".format(epoch+1, num_epochs)):

                inputs = batch[0].to(device) # This moves the input data batch to the specific device(CPU or GPU)
                optimizer.zero_grad()
                outputs = self(inputs) # Forward pass
                loss = criterion(outputs.view(-1, self.vocab_size), inputs.view(-1))  # Calculate loss
                loss.backward() # Backward pass
                optimizer.step()
                running_loss += loss.item()

            epoch_loss = running_loss / len(train_dataloader)
            print("Epoch {} Loss: {:.4f}".format(epoch+1, epoch_loss))


if __name__ == "__main__":

    train_data = pd.read_csv("./corpus/train.csv")
    train_tokens = train_data['Description'][:10000].apply(word_tokenize).tolist() # modify it later 
    
    all_tokens = [token for sublist in train_tokens for token in sublist]
    word_counts = Counter(all_tokens)
    vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    vocab_size = len(vocab)
    words_to_ind = {word: index for index, word in enumerate(vocab)}

    train_tokens_indices = [[words_to_ind[token] for token in tokens] for tokens in train_tokens]

    elmo = ELMO(vocab_size, 300, 100, 64, words_to_ind)
    elmo.train(train_tokens_indices)
    
    embeddings = elmo.embedding.weight.cpu().detach().numpy()

    data_to_store = {
        "model": elmo, # This itself contain word to ind mapping using which we can access the embeddings (also stored in this dict itself) by embeddings[idx]
        "embeddings": embeddings
    }
    torch.save(data_to_store, 'bilstm.pt')