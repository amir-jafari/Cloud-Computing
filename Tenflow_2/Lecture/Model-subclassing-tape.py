import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
p = np.linspace(-2, 2, 100)
t = np.sin(np.pi*p)
p = tf.reshape(tf.convert_to_tensor(p), (-1, 1))
t = tf.reshape(tf.convert_to_tensor(t), (-1, 1))
class MyModel(tf.keras.Model):
  def __init__(self):
    super(MyModel, self).__init__()
    inputs = tf.keras.Input(shape=(1))
    self.x1 = tf.keras.layers.Dense(512, activation='tanh', name='d1')
    self.predictions = tf.keras.layers.Dense(1, name='d2')

  def call(self, inputs):
    x = self.x1(inputs)
    return self.predictions(x)
model = MyModel()

model = MyModel()
optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
criterion = tf.keras.losses.MeanSquaredError()

# %% ----- Training Loop -------------
N_EPOCHS = 10000
PRINT_LOSS_EVERY = 100
for epoch in range(N_EPOCHS):
    with tf.GradientTape() as tape:
        t_pred = model(p)
        loss = criterion(t, t_pred)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    if epoch % PRINT_LOSS_EVERY == 0:
        print("Epoch {} | Loss {:.5f}".format(
            epoch, tf.reduce_mean(loss).numpy()))

pred = model(p)

plt.xlabel("x")
plt.ylabel("y")
plt.plot(p, t, label="Real Function")
plt.plot(p, pred, linestyle="dashed", label="MLP Approximation")
plt.show()