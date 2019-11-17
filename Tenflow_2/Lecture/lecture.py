import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
p = np.linspace(-2, 2, 100)
t = np.sin(np.pi*p)

inputs = tf.keras.Input(shape=(1))
x = tf.keras.layers.Dense(10, activation='relu')(inputs)
predictions = tf.keras.layers.Dense(1)(x)

model = tf.keras.Model(inputs=inputs, outputs=predictions)
print(model.summary())

optimiser = tf.keras.optimizers.Adam()
model.compile(optimizer=optimiser,loss='mse')

model.fit(p, t, epochs=5000,batch_size=10, verbose=1 )
pred = model.predict(p)

plt.xlabel("x")
plt.ylabel("y")
plt.plot(p, t, label="Real Function")
plt.plot(p, pred, linestyle="dashed", label="MLP Approximation")
plt.show()
# # %%----------------------------------------------------------------------------------
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
p = np.linspace(-2, 2, 100)
t = np.sin(np.pi*p)

model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(10, activation='tanh'),
  tf.keras.layers.Dense(1, activation='linear')
])

model.compile(optimizer='adam',loss='mse')
model.fit(p, t, epochs=5000,batch_size=10, verbose=1 )
pred = model.predict(p)

plt.xlabel("x")
plt.ylabel("y")
plt.plot(p, t, label="Real Function")
plt.plot(p, pred, linestyle="dashed", label="MLP Approximation")
plt.show()
# %%----------------------------------------------------------------------------------
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
# %%----------------------------------------------------------------------------------
import tensorflow as tf

mnist =  tf.keras.datasets.mnist
(train_x,train_y), (test_x, test_y) = mnist.load_data()

train_x, test_x = tf.cast(train_x/255.0, tf.float32), tf.cast(test_x/255.0, tf.float32)
train_y, test_y = tf.cast(train_y,tf.int64),tf.cast(test_y,tf.int64)

epochs=10
batch_size = 32

train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y)).batch(32).shuffle(10000)
train_dataset = train_dataset.map(lambda x, y: (tf.image.random_flip_left_right(x), y))
train_dataset = train_dataset.repeat()

test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y)).batch(batch_size).shuffle(10000)
test_dataset = train_dataset.repeat()

model = tf.keras.models.Sequential([
 tf.keras.layers.Flatten(),
 tf.keras.layers.Dense(512,activation=tf.nn.relu),
 tf.keras.layers.Dropout(0.2),
 tf.keras.layers.Dense(10,activation=tf.nn.softmax)
])

#required becuase of the repeat() on the dataset
steps_per_epoch = len(train_x)//batch_size

optimiser = tf.keras.optimizers.Adam()
model.compile (optimizer= optimiser, loss='sparse_categorical_crossentropy', metrics = ['accuracy'])

model.fit(train_dataset, epochs=epochs, steps_per_epoch = steps_per_epoch)
model.evaluate(test_dataset,steps=10)

model.fit(train_dataset, epochs=epochs, steps_per_epoch=steps_per_epoch,
          validation_data=test_dataset,
          validation_steps=3)
model.evaluate(test_dataset,steps=10)