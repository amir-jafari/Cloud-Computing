## Example: 1D-CNN for Sentiment Analysis

This example builds on `../MLP/4_TextClassification` by replacing the MLP with a 1D-CNN. This works by having a kernel of size `embed_dim` x `n_gram` slide across `n_gram` words at the same time. Although 1D-CNNs make more sense more signal data, this does improve results compared to the MLP.

### New Code

- `nn.Conv1d` and permuting the output of the embedding (each embedding dim is treated as an input chanel to the CNN).
- `tqdm` for training progress bars.

## Exercise: 1D-CNN for Paraphrase Identification

The [Microsoft Research Paraphrase Corpus](https://www.microsoft.com/en-us/download/details.aspx?id=52398) is a dataset containing pairs of sentences with labels telling whether they are paraphrases or not (whether they mean the same).


### New Code

- Handling sentence-pair nlp tasks.
- Reading raw `.txt` and handling lines with problems. 
