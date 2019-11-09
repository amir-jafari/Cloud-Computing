## Example: LSTM for Signal Forecasting

This example shows how to use a LSTM to predict the next value of a [chirp signal](https://en.wikipedia.org/wiki/Chirp) using the previous `SEQ_LENTH` values.

### New Code

- Preparing input data for LSTM into suitable format, including batching.
- Splitting into training and testing for time-series data.
- LSTM layer.
- LSTM training loop.
- Stateful vs Stateless.
- Visualizing the predictions made by the LSTM.


## Exercise: GRU for Signal Forecasting

This exercise asks to use a GRU instead of a LSTM for the same problem.

### New Code

- GRU code equivalent to the one from the example.
