## Example: MLP for Sentiment Analysis

The [Stanford Sentiment Treebank](https://nlp.stanford.edu/sentiment/treebank.html) is a dataset containing movie reviews labels as either positive or negative. While a MLP is clearly not ideal given the temporal nature of language, this example illustrates how to get a dictionary of unique token ids and how to use this token ids as input to an Embedding layer. More advanced example using RNNs and Attention will build on top of this.

### New Code

- Getting a vocabulary of unique token ids and a maximum sequence length.
- Zero-padding sequeces shorter than this length.
- Using these ids as inputs to `nn.Embedding`.
- Saving files so that we can focus on the model after all the preprocessing. 

## Exercise: MLP for Sentiment Analysis with Pretrained Word Embeddings

This exercise asks to use the [glove word embeddings](https://nlp.stanford.edu/projects/glove/) instead of learning them from scratch. In order to do so, a good understanding of PyTorch's parameters and tensors is needed.

### New Code

- Loading pretrained word embeddings.
- Freezing embeddings (and weights in general).
