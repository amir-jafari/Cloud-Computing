## Example: Data Pipelines

This example illustrates some of the functionalities that `tf.Data` provides to create your own datasets and train with them, instead of plain `tf.Tensor`s. We will be using an image classification problem consisting of [102 distinct flowers](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html). Apart from the useful methods like batching and shuffling, the most important thing is that this allows to only load each batch into GPU memory, which is totally neccessary when you are working with big data.

### New Code

- Splitting into training and testing datasets where each example is an individual file.
- Basic use of `tf.Data` to manage data pipelines, shuffling and batching during training.

## Exercise: Custom Data Pipelines 

While the basic use of `tf.data` is pretty useful, there are sometimes when you need special pre-processing that is not avaibale in TensorFlow's built-in functions. Thankfully, we can still create a TensorFlow Data Pipeline using generators.

### New Code

- Python generators.
- Using `tf.data.Dataset.from_generator`. 
