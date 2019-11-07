## Example: 1D-CNN for Speech Recognition

`example_SpokenDigitRecognizer.py` trains a 1D-CNN to classify audio files of [spoken digits](https://github.com/Jakobovski/free-spoken-digit-dataset) (from 0 to 9). An audio file is basically an audio signal encoded in digital format. An audio signal is nothing more than a [sound wave](https://en.wikipedia.org/wiki/Acoustic_wave_equation). The physics can range from quite simple to very complex, but you can think of it as a sinusoidal wave $p = p_0 sin(\omega t + kx)$. $\omega$ is the angular frequency, $k$ is the wave number, $t$ is time and $x$ is position (in 1-D). $p$ is the intensity and $p_0$ is the amplitud.  We are not concerned with space in this case, so let's just think of it as $p = p_0 sin(\omega t)$. This is a continuous equation, but computers work in a discrete domain, so we just store a number of samples for the value of $p$ per each second. This n_samples/second is called sampling frequency.

We use 1D convolution and pooling layers to learn local features across this time dimension, in order to classify the spoken digits.

### New Code

- 1D convolutions and pooling.
- Progress bars with `tqdm`.
- Save and load `tf.keras.Model` subclassed classes. [This](https://colab.research.google.com/drive/172D4jishSgE3N7AO6U2OKAA_0wNnrMOq#scrollTo=OOSGiSkHTERy) is a good guide for saving models in general in TensorFlow 2.


## Exercise: Recognize self-recorded spoken digits

This exercise asks to use the model trained on the example to recognize audio recorded by me or by you. If you want to record your own audio, use [Audacity](https://www.audacityteam.org/download/) and select "Project Rate (Hz)" = 8000, and "1 (Mono) Recording Channel".

### New Code

- Loading a model and useing it only for inference.
- Using `tf.nn.softmax` to get the probabilities from output logits.







