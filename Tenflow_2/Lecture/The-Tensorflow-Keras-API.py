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