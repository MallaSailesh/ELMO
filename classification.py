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
from elmo import ELMO
from sklearn.metrics import confusion_matrix, classification_report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ELMOClassifier(nn.Module):

    def __init__(self, elmo_model, elmo_embedding, num_classes):

        super(ELMOClassifier, self).__init__()

        # self.lambdas = nn.Parameter(torch.rand(3)) # -> 4.1      
        self.lambdas = [0.33, 0.33, 0.33] # -> 4.2 
        
        self.elmo = elmo_model
        self.embedding = elmo_embedding
        self.num_classes = num_classes
        self.fc = nn.Linear(elmo_model.hidden_dim*2, num_classes)  
        self.linear_embed = nn.Linear(elmo_model.embedding_dim, elmo_model.hidden_dim*2) 

        # Define the combination function - here this is a neural network
        self.combine_layer = nn.Sequential(
            nn.Linear(elmo_model.hidden_dim * 2 * 3, elmo_model.hidden_dim * 2),
            nn.ReLU()
        )  # 4.3

    def forward(self, x):
        x_embed = torch.tensor(self.embedding[x]).float()
        x_embed_transformed = self.linear_embed(x_embed) 
        x_lstm1, _ = self.elmo.lstm1(x_embed)
        x_lstm2, _ = self.elmo.lstm2(x_lstm1)

        # combined = (self.lambdas[0]*x_embed_transformed  + self.lambdas[1]*x_lstm1 +self.lambdas[2]*x_lstm2)/(self.lambdas[0]+self.lambdas[1] + self.lambdas[2])  # 4.1 , 4.2 

        combined = torch.cat((x_embed_transformed, x_lstm1, x_lstm2), dim=-1) # Concatenate the embeddings from different layers -> 4.3
        combined = self.combine_layer(combined) # 4.3

        combined = torch.max(combined, dim=1)[0] 

        logits = self.fc(combined)
        return logits

    def train(self, x_train, y_train, batch_size=1, epochs=5, lr=0.001):

        max_length = max(len(seq) for seq in x_train) # Determine the maximum length of sequences
        padded_sequences = [torch.tensor(seq + [0] * (max_length - len(seq)), dtype=torch.long) for seq in x_train] # Pad sequences to the maximum length
        x_train = torch.stack(padded_sequences) # Stack padded sequences into a tensor

        y_train = torch.tensor(y_train, dtype=torch.long) # Convert data into tensors

        train_dataset = TensorDataset(x_train, y_train) # Create a TensorDataset
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # Create a DataLoader

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)

        self.to(device) # Move model to device

        for epoch in range(epochs):
            total_loss = 0.0
            total_correct = 0

            for batch_x, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                
                batch_x, batch_y = batch_x.to(device), batch_y.to(device) # Move data to device
                optimizer.zero_grad() # Zero the gradients
                outputs = self.forward(batch_x) # Forward pass
    
                batch_y -= 1

                loss = criterion(outputs, batch_y)
                loss.backward() # Backward pass
                optimizer.step()
                total_loss += loss.item() 

                # Calculate total correct predictions
                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == batch_y).sum().item()

            epoch_loss = total_loss / len(train_loader.dataset)
            epoch_accuracy = total_correct / len(train_loader.dataset)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")     

    def eval(self, x_eval, y_eval, batch_size=64):

        max_length = max(len(seq) for seq in x_eval) # Determine the maximum length of sequences
        padded_sequences = [torch.tensor(seq + [0] * (max_length - len(seq)), dtype=torch.long) for seq in x_eval] # Pad sequences to the maximum length
        x_eval = torch.stack(padded_sequences) # Stack padded sequences into a tensor

        y_eval = torch.tensor(y_eval, dtype=torch.long)

        eval_dataset = TensorDataset(x_eval, y_eval) 
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True) 

        self.to(device)

        all_predictions = []
        all_labels = []

        for batch_x, batch_y in tqdm(eval_loader, desc="Evaluating"):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            batch_y -= 1
            outputs = self.forward(batch_x)
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

        cm = confusion_matrix(all_labels, all_predictions)
        report = classification_report(all_labels, all_predictions, zero_division=1)
        print("Confusion Matrix:")
        print(cm)
        print("\nClassification Report:")
        print(report)


if __name__ == "__main__":

    elmo = torch.load('bilstm.pt')
    elmo_model = elmo["model"]
    elmo_embeddings = elmo["embeddings"]
    words_to_ind = elmo_model.words_to_ind

    train_data = pd.read_csv("./corpus/train.csv")
    test_data = pd.read_csv("./corpus/test.csv")

    train_tokens = train_data['Description'][:10000].apply(word_tokenize).tolist() # modify it later 
    test_tokens = test_data['Description'][:].apply(word_tokenize).tolist() # modify it later 
    train_labels = train_data['Class Index'][:10000].tolist()
    test_labels = test_data['Class Index'][:].tolist()

    train_tokens_indices = [[words_to_ind[token] for token in tokens] for tokens in train_tokens]
    test_tokens_indices = [
        [words_to_ind[token] if token in words_to_ind else elmo_model.vocab_size for token in tokens]
        for tokens in test_tokens
    ]

    elmo_embeddings = np.append(elmo_embeddings, np.zeros((1, elmo_model.embedding_dim)), axis=0)

    elmo_classifier = ELMOClassifier(elmo_model, elmo_embeddings, 4)
    elmo_classifier.train(train_tokens_indices, train_labels)
    print("\n\t Evaluation on Train Data \n")
    elmo_classifier.eval(train_tokens_indices, train_labels)
    print("\n\t Evaluation on Test Data \n")
    elmo_classifier.eval(test_tokens_indices, test_labels)

    # 4.1
    # print("\nTrained Lambdas:\n")
    # for i in elmo_classifier.lambdas:
    #     print(i)

    torch.save(elmo_classifier, 'classifier_3.pt') 

