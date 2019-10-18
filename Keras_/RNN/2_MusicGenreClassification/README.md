## Example: Musical Genre Classification

This example trains a CNN-LSTM to classify raw audio files of songs into 10 different genres. We use several 1D-CNN and 1D-Pooling layers to learn local features and reduce the sequence length (too long to input on the LSTM raw, as the examples are 30 seconds long and the sampling frequency is 22050 Hz). 

### New Code

- Reading and preprocessing `.wav` files
- CNN + LSTM
- Stacking multiple LSTM layers

## Exercise: Listening to the 1D-CNNs outputs 

If we can get the output audio signals of the CNN layers and listen to them... why not do it? 

### New Code

- Naming layers and accessing intermidiate outputs.
- Saving NumPy arrays to `.wav` files.
