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