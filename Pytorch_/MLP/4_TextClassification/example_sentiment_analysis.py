# Download the Stanford Sentiment Treebank from https://gluebenchmark.com/tasks and unzip it in the current working dir

# %% --------------------------------------- Imports -------------------------------------------------------------------
import os
import numpy as np
import pandas as pd
import json
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix
import nltk
nltk.download('punkt')

# %% --------------------------------------- Set-Up --------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# %% ----------------------------------- Hyper Parameters --------------------------------------------------------------
class Args:
    def __init__(self):
        self.seq_len = "get_max_from_data"
        # self.seq_len = 30
        self.embedding_dim = 100
        self.n_neurons = (100, 200, 100)
        self.n_epochs = 10
        self.lr = 1e-2
        self.batch_size = 512
        self.dropout = 0.2
        self.train = True
        self.save_model = True


args = Args()


# %% ----------------------------------- Helper Functions --------------------------------------------------------------
def acc(x, y, return_labels=False):
    with torch.no_grad():
        logits = model(x)
        pred_labels = np.argmax(logits.cpu().numpy(), axis=1)
    if return_labels:
        return pred_labels
    else:
        return 100*accuracy_score(y.cpu().numpy(), pred_labels)


def extract_vocab_dict_and_msl(sentences_train, sentences_dev):
    """ Tokenizes all the sentences and gets a dictionary of unique tokens and also the maximum sequence length """
    tokens, ms_len = [], 0
    for sentence in list(sentences_train) + list(sentences_dev):
        tokens_in_sentence = nltk.word_tokenize(sentence)
        if ms_len < len(tokens_in_sentence):
            ms_len = len(tokens_in_sentence)
        tokens += tokens_in_sentence
    # We reserve the 0 id for padded 0s
    token_vocab = {key: i for key, i in zip(set(tokens), range(1, len(set(tokens))+1))}
    if len(np.unique(list(token_vocab.values()))) != len(token_vocab):
        "There are some rep words..."
    return token_vocab, ms_len


def convert_to_ids(raw_sentences, vocab_dict, pad_to):
    """ Takes an NumPy array of raw text sentences and converts to a sequence of token ids """
    x = np.empty((len(raw_sentences), pad_to))
    for idx, sentence in enumerate(raw_sentences):
        word_ids = []
        for token in nltk.word_tokenize(sentence):
            try:
                word_ids.append(vocab_dict[token])
            except:  # This option is to handle out-of-vocab words, which will be assigned the same token id. There are
                print("Unknown token encountered:", token)  # better ways of handling this, like WordPiece embeddings.
                word_ids.append(len(vocab_dict))  # Adds one more unique id which will stand for unknown tokens
        if pad_to < len(word_ids):
            x[idx] = word_ids[:pad_to]
        else:
            x[idx] = word_ids + [0] * (pad_to - len(word_ids))
    return x


# %% -------------------------------------- MLP Class ------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, vocab_size, neurons_per_layer):
        super(MLP, self).__init__()
        # Maps token ids (integers) to a embedding_dim dimensional space (vocab_size+2 to account for unknown and padded tokens)
        self.embedding = nn.Embedding(vocab_size+2, args.embedding_dim)
        # MLP part, the input dim to the first layer must be seq_len * embedding_dim
        dims = (args.seq_len*args.embedding_dim, *neurons_per_layer)
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dims[i], dims[i+1]),
                nn.ReLU(),
                nn.BatchNorm1d(dims[i+1]),
                nn.Dropout(args.dropout)
            ) for i in range(len(dims)-1)
        ])
        self.layers.extend(nn.ModuleList([nn.Linear(neurons_per_layer[-1], 2)]))

    def forward(self, x):
        x = self.embedding(x)
        x = x.reshape(x.shape[0], -1)  # Flattens the input to (batch_size, seq_len * embedding_dim), to use all the
        for layer in self.layers:  # embedding components of all the words as the features of the input sequence
            x = layer(x)
        return x


# %% -------------------------------------- Data Prep ------------------------------------------------------------------
# Loads all the data
data_train = pd.read_csv("SST-2/train.tsv", sep="\t")
x_train_raw, y_train = data_train["sentence"].values, torch.LongTensor(data_train["label"].values).to(device)
data_dev = pd.read_csv("SST-2/dev.tsv", sep="\t")
x_dev_raw, y_dev = data_dev["sentence"].values, torch.LongTensor(data_dev["label"].values).to(device)

try:  # Tries to open the vocab dict and the maximum sequence length
    with open("example_prep_data/vocab_dict.json", "r") as s:
        token_ids = json.load(s)
    msl = np.load("example_prep_data/max_sequence_length.npy").item()
except:  # If it fails, gets them from the corpus and saves them
    print("Tokenizing all the examples to get a vocab dict and the maximum sequence length...")
    token_ids, msl = extract_vocab_dict_and_msl(x_train_raw, x_dev_raw)
    # Saves the dict to json so that we can just load it the next time
    os.mkdir("example_prep_data")
    with open("example_prep_data/vocab_dict.json", "w") as s:
        json.dump(token_ids, s)
    np.save("example_prep_data/max_sequence_length.npy", np.array([msl]))  # Saves msl to numpy
if args.seq_len == "get_max_from_data":
    args.seq_len = msl
del data_train, data_dev  # Deletes the variables we don't need anymore

# Loads or tokenizes all the sentences and converts to the token ids using the vocab dict
try:
    x_train = np.load("example_prep_data/prep_train_len{}.npy".format(args.seq_len))
    x_dev = np.load("example_prep_data/prep_dev_len{}.npy".format(args.seq_len))
except:
    print("Converting all the sentences to sequences of token ids...")
    x_train = convert_to_ids(x_train_raw, token_ids, args.seq_len)
    np.save("example_prep_data/prep_train_len{}.npy".format(args.seq_len), x_train)
    x_dev = convert_to_ids(x_dev_raw, token_ids, args.seq_len)
    np.save("example_prep_data/prep_dev_len{}.npy".format(args.seq_len), x_dev)
del x_train_raw, x_dev_raw  # Deletes the variables we don't need anymore

x_train, x_dev = torch.LongTensor(x_train).to(device), torch.LongTensor(x_dev).to(device)
# In this case we don't set x_train.requires_grad = True because the embedding layer acts as a trainable look-up table,
# i.e, the output of the embedding layer has grad_fn=EmbeddingBackward, so the weight of the embedding
# layer is being updated by the gradient computed from this function when going backwards, instead of the usual
# grad_fn=AddmmBackward. The difference is that instead of doing matrix-multiplication, a look up is more efficient
# because we have 1 row of the model.embedding.weight for each id in our vocabulary. The rest of the multiplications
# would be by 0s if we used a one-hot-encoded input vector instead of a token id. Mathematically, however,
# it's the exact same weight update.

# %% -------------------------------------- Training Prep ----------------------------------------------------------
model = MLP(len(token_ids), args.n_neurons).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
criterion = nn.CrossEntropyLoss()

# %% -------------------------------------- Training Loop ----------------------------------------------------------
labels_ditrib = torch.unique(y_dev, return_counts=True)
print("The no information rate is {:.2f}".format(100*labels_ditrib[1].max().item()/len(y_dev)))
if args.train:
    acc_dev_best = 0
    print("Starting training loop...")
    for epoch in range(args.n_epochs):

        loss_train = 0
        model.train()
        for batch in range(len(x_train)//args.batch_size + 1):
            inds = slice(batch*args.batch_size, (batch+1)*args.batch_size)
            optimizer.zero_grad()
            logits = model(x_train[inds])
            loss = criterion(logits, y_train[inds])
            loss.backward()
            optimizer.step()
            loss_train += loss.item()

        model.eval()
        with torch.no_grad():
            y_dev_pred = model(x_dev)
            loss = criterion(y_dev_pred, y_dev)
            loss_test = loss.item()

        acc_dev = acc(x_dev, y_dev)
        print("Epoch {} | Train Loss {:.5f}, Train Acc {:.2f} - Test Loss {:.5f}, Test Acc {:.2f}".format(
            epoch, loss_train/args.batch_size, acc(x_train, y_train), loss_test, acc_dev))

        if acc_dev > acc_dev_best and args.save_model:
            torch.save(model.state_dict(), "mlp_sentiment.pt")
            print("The model has been saved!")
            acc_dev_best = acc_dev

# %% ------------------------------------------ Final test -------------------------------------------------------------
model.load_state_dict(torch.load("mlp_sentiment.pt"))
model.eval()
y_test_pred = acc(x_dev, y_dev, return_labels=True)
print("The accuracy on the test set is {:.2f}".format(100*accuracy_score(y_dev.cpu().numpy(), y_test_pred), "%"))
print("The confusion matrix is")
print(confusion_matrix(y_dev.cpu().numpy(), y_test_pred))
