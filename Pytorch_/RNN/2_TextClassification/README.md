## Example: LSTM for Sentiment Analysis

This example builds on `../MLP/4_TextClassification` by replacing the MLP with three LSTMs. The LSTMs process all the words on the input sequence recursively and output some features for each word. These features are weighted over time by a learnable linear layer and an output linear layer maps this average to the classification space.

### New Code

- Modeling text via LSTM.
- Using `torch.nn.utils.rnn.PackedSequence` to omit the zero-padded tokens and still have vectorized mini-batching.

## Exercise: BiLSTMs for Sentiment Analysis

The exercise build on the example by asking to use Bidirectional LSTMs to incorporate future context into the processing of each word.


### New Code

- Using BiLSTMs including different of combining their outputs.
