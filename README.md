# ELMo: Deep Contextualized Word Representations

## Introduction
Modern word embedding algorithms like word2vec and GloVe provide single representations for words, ignoring contextual information. ELMo, a contextualized embedding model, addresses this by capturing word meaning in context using stacked Bi-LSTM layers. This README outlines the implementation and training of an ELMo architecture from scratch using PyTorch.

## Implementation and Training

### Architecture
The ELMo architecture consists of stacked Bi-LSTM layers to generate contextualized word embeddings. Weights for combining word representations across layers are trained.

### Model Pre-training
ELMo embeddings are learned through bidirectional language modeling on the given dataset's train split.

- Trained model: `bilstm.pt`
- [Download Model](https://drive.google.com/file/d/1DoSxvG_UWxeeSNhpA056FGDv4LfnUJkT/view?usp=sharing)

### Downstream Task
Trained the ELMo architecture on a 4-way classification task using the AG News Classification Dataset.

## Corpus
Trained the model on the provided News Classification Dataset (same dataset used for other methods in getting word embeddingd - check Word_Vectorization repository of mine for more detail).

- [Download Corpus](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/advaith_malladi_research_iiit_ac_in/EWjgIboHC19Ppq6Of9klUo4BlKgAqynxC0TRBURzQ0lEzA?e=tWZqY5)

## Hyperparameter Tuning

### Trainable λs
Trained and found the best λs for combining word representations across different layers.

- Model: `classifier_1.pt`
- [Download Model](https://drive.google.com/file/d/1GCG-1N3rOpAsDpdrziDz0xN-hGWqbVui/view?usp=sharing)

### Frozen λs
Randomly initialized and froze the λs.

- Model: `classifier_2.pt`
- [Download Model](https://drive.google.com/file/d/1W2MRBK9rk57UM5-69RbbFZgke4vHt-M7/view?usp=sharing)

### Learnable Function
Learned a function to combine word representations across layers.

- Model: `classifier_3.pt`
- [Download Model](https://drive.google.com/file/d/1h5tabWE2zlHHDdECD69dFdeTcBYKyldF/view?usp=sharing)

## Analysis
Comprehensive analysis of ELMo's performance in pretraining and the downstream task compared to SVD and Word2Vec embeddings. Included performance metrics like accuracy, F1 score, precision, recall, and confusion matrices for different settings.

### Loading Models
#### Load Model
```python
data = torch.load("<filename>")
```

#### To load any model ( .pt files ) :-
```python
`<data retrieved>` = torch.load("`<filename>`")
```

Note :- 
- While pretraining the elmo , i used only first 10000 sentences in train.csv for it
- Also i used only first 10000 train sentences for downstream task also