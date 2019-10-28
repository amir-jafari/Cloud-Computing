# Download the Stanford Sentiment Treebank from https://gluebenchmark.com/tasks and unzip it in the current working dir
# Download glove.6B.zip from https://nlp.stanford.edu/projects/glove/, unzip it and move glove.6B.50d.txt to the
# current working directory.

# %% --------------------------------------- Imports -------------------------------------------------------------------
import os
import numpy as np
import pandas as pd
import json
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix
import nltk
from tqdm import tqdm
nltk.download('punkt')

if "SST-2" not in os.listdir(os.getcwd()):
    try:
        os.system("wget https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FSST-2.zip?alt=media&token=aabc5f6b-e466-44a2-b9b4-cf6337f84ac8")
        os.system("unzip SST-2.zip")
    except:
        print("There was a problem with the download!")
        # Download the Stanford Sentiment Treebank from https://gluebenchmark.com/tasks and unzip it in the current working dir
    if "SST-2" not in os.listdir(os.getcwd()):
        print("There was a problem with the download!")
        import sys
        sys.exit()
if "glove.6B.50d.txt" not in os.listdir(os.getcwd()):
    try:
        os.system("wget http://nlp.stanford.edu/data/glove.6B.zip")
        os.system("unzip glove.6B.zip")
        os.system("mv glove.6B/glove.6B.50d.txt glove.6B.50d.txt")
        os.system("sudo rm -r glove.6B")
    except:
        print("There as a problem downloading the data!")
        raise
    if "glove.6B.50d.txt" not in os.listdir(os.getcwd()):
        print("There as a problem downloading the data!")
        # Download glove.6B.zip from https://nlp.stanford.edu/projects/glove/, unzip it and move glove.6B.50d.txt to the
        # current working directory.

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
        self.embedding_dim = 50
        self.n_epochs = 40
        self.lr = 1e-2
        self.batch_size = 512
        self.train = True
        self.save_model = True


args = Args()


# %% ----------------------------------- Helper Functions --------------------------------------------------------------
def acc(x, y, return_labels=False):
    with torch.no_grad():
        logits = torch.empty(len(x), 2)
        for batch in range(len(x) // args.batch_size + 1):
            inds = slice(batch * args.batch_size, (batch + 1) * args.batch_size)
            logits[inds] = model(x[inds])
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
            except:
                word_ids.append(vocab_dict[token])
        if pad_to < len(word_ids):
            x[idx] = word_ids[:pad_to]
        else:
            x[idx] = word_ids + [0] * (pad_to - len(word_ids))
    return x


def get_glove_embeddings(vocab_dict):
    with open("glove.6B.50d.txt", "r") as s:
        glove = s.read()
    embeddings_dict = {}
    for line in glove.split("\n")[:-1]:
        text = line.split()
        if text[0] in vocab_dict:
            embeddings_dict[vocab_dict[text[0]]] = torch.from_numpy(np.array(text[1:], dtype="float32"))
    return embeddings_dict


def get_glove_table(vocab_dict, glove_dict):
    lookup_table = torch.empty((len(vocab_dict)+2, 50))
    for token_id in sorted(vocab_dict.values()):
        if token_id in glove_dict:
            lookup_table[token_id] = glove_dict[token_id]
        else:
            lookup_table[token_id] = torch.zeros((1, 50))  # For unknown tokens
    lookup_table[0] = torch.zeros((1, 50))
    return lookup_table


# %% -------------------------------------- CNN Class ------------------------------------------------------------------
class CNN(nn.Module):
    def __init__(self, vocab_size):
        super(CNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size + 2, args.embedding_dim)

        self.conv1 = nn.Conv1d(args.embedding_dim, args.embedding_dim, 9)
        self.convnorm1 = nn.BatchNorm1d(args.embedding_dim)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(args.embedding_dim, args.embedding_dim, 9)
        self.convnorm2 = nn.BatchNorm1d(args.embedding_dim)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(args.embedding_dim, args.embedding_dim, 7)
        self.linear = nn.Linear(args.embedding_dim, 2)
        self.act = torch.relu

    def forward(self, x):
        # nn.Conv1d operates on the columns, each embedding dimension is considered as one channel
        x = self.embedding(x).permute(0, 2, 1)
        x = self.pool1(self.convnorm1(self.act(self.conv1(x))))
        x = self.pool2(self.convnorm2(self.act(self.conv2(x))))
        return self.linear(self.act(self.conv3(x)).reshape(-1, args.embedding_dim))


# %% -------------------------------------- Data Prep ------------------------------------------------------------------
data_train = pd.read_csv("SST-2/train.tsv", sep="\t")
x_train_raw, y_train = data_train["sentence"].values, torch.LongTensor(data_train["label"].values).to(device)
data_dev = pd.read_csv("SST-2/dev.tsv", sep="\t")
x_dev_raw, y_dev = data_dev["sentence"].values, torch.LongTensor(data_dev["label"].values).to(device)

try:
    with open("example_prep_data/vocab_dict.json", "r") as s:
        token_ids = json.load(s)
    msl = np.load("example_prep_data/max_sequence_length.npy").item()
except:
    print("Tokenizing all the examples to get a vocab dict and the maximum sequence length...")
    token_ids, msl = extract_vocab_dict_and_msl(x_train_raw, x_dev_raw)
    os.mkdir("example_prep_data")
    with open("example_prep_data/vocab_dict.json", "w") as s:
        json.dump(token_ids, s)
    np.save("example_prep_data/max_sequence_length.npy", np.array([msl]))
if args.seq_len == "get_max_from_data":
    args.seq_len = msl
del data_train, data_dev

glove_embeddings = get_glove_embeddings(token_ids)

try:
    x_train = np.load("example_prep_data/prep_train_len{}.npy".format(args.seq_len))
    x_dev = np.load("example_prep_data/prep_dev_len{}.npy".format(args.seq_len))
except:
    print("Converting all the sentences to sequences of token ids...")
    x_train = convert_to_ids(x_train_raw, token_ids, args.seq_len)
    np.save("example_prep_data/prep_train_len{}.npy".format(args.seq_len), x_train)
    x_dev = convert_to_ids(x_dev_raw, token_ids, args.seq_len)
    np.save("example_prep_data/prep_dev_len{}.npy".format(args.seq_len), x_dev)
del x_train_raw, x_dev_raw

x_train, x_dev = torch.LongTensor(x_train).to(device), torch.LongTensor(x_dev).to(device)

# %% -------------------------------------- Training Prep ----------------------------------------------------------
model = CNN(len(token_ids)).to(device)
look_up_table = get_glove_table(token_ids, glove_embeddings)
model.embedding.weight.data.copy_(look_up_table)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
criterion = nn.CrossEntropyLoss()

# %% -------------------------------------- Training Loop ----------------------------------------------------------
labels_ditrib = torch.unique(y_dev, return_counts=True)
print("The no information rate is {:.2f}".format(100*labels_ditrib[1].max().item()/len(y_dev)))
if args.train:
    acc_dev_best = 0
    print("Starting training loop...")
    for epoch in range(args.n_epochs):

        loss_train, train_steps = 0, 0
        model.train()
        total = len(x_train) // args.batch_size + 1  # Initiates a progress bar that will be updated for each batch
        with tqdm(total=total, desc="Epoch {}".format(epoch)) as pbar:  # "Epoch" will be updated for each epoch
            for batch in range(len(x_train)//args.batch_size + 1):
                inds = slice(batch*args.batch_size, (batch+1)*args.batch_size)
                optimizer.zero_grad()
                logits = model(x_train[inds])
                loss = criterion(logits, y_train[inds])
                loss.backward()
                optimizer.step()
                loss_train += loss.item()
                train_steps += 1
                pbar.update(1)  # Updates the progress and the training loss
                pbar.set_postfix_str("Training Loss: {:.5f}".format(loss_train / train_steps))

        model.eval()
        with torch.no_grad():
            y_dev_pred = model(x_dev)
            loss = criterion(y_dev_pred, y_dev)
            loss_test = loss.item()

        acc_dev = acc(x_dev, y_dev)
        print("Epoch {} | Train Loss {:.5f}, Train Acc {:.2f} - Test Loss {:.5f}, Test Acc {:.2f}".format(
            epoch, loss_train/train_steps, acc(x_train, y_train), loss_test, acc_dev))

        if acc_dev > acc_dev_best and args.save_model:
            torch.save(model.state_dict(), "cnn_sentiment.pt")
            print("The model has been saved!")
            acc_dev_best = acc_dev

# %% ------------------------------------------ Final test -------------------------------------------------------------
model.load_state_dict(torch.load("cnn_sentiment.pt"))
model.eval()
y_test_pred = acc(x_dev, y_dev, return_labels=True)
print("The accuracy on the test set is {:.2f}".format(100*accuracy_score(y_dev.cpu().numpy(), y_test_pred), "%"))
print("The confusion matrix is")
print(confusion_matrix(y_dev.cpu().numpy(), y_test_pred))
