import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
p = np.linspace(-2, 2, 100)
t = np.sin(np.pi*p)


class MyModel(tf.keras.Model):
  def __init__(self):
    super(MyModel, self).__init__()
    inputs = tf.keras.Input(shape=(1))
    self.x1 = tf.keras.layers.Dense(512, activation='relu', name='d1')
    self.predictions = tf.keras.layers.Dense(1, name='d2')

  def call(self, inputs):
    x = self.x1(inputs)
    return self.predictions(x)
model = MyModel()

optimiser = tf.keras.optimizers.Adam()
model.compile (optimizer= optimiser, loss='mse')
model.fit(p, t, batch_size=32, epochs=1000)
pred = model.predict(p)

plt.xlabel("x")
plt.ylabel("y")
plt.plot(p, t, label="Real Function")
plt.plot(p, pred, linestyle="dashed", label="MLP Approximation")
plt.show()