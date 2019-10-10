import tensorflow as tf
#tf.set_random_seed(1)
# ----------------------------------------------------------------------------------
x1_data = [1, 2, 3, 4, 5]
y_data = [2, 3, 4, 5, 6]
# ----------------------------------------------------------------------------------
x1 = tf.placeholder(tf.float32)

Y = tf.placeholder(tf.float32)
# ----------------------------------------------------------------------------------
w1 = tf.Variable(tf.random_normal([1]), name='weight1')
b = tf.Variable(tf.random_normal([1]), name='bias')
# ----------------------------------------------------------------------------------
hypothesis = x1 * w1  + b
print(hypothesis)
# ----------------------------------------------------------------------------------
cost = tf.reduce_mean(tf.square(hypothesis - Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-2)
train = optimizer.minimize(cost)
# ----------------------------------------------------------------------------------
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(5000):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={x1: x1_data,  Y: y_data})
    if step % 10 == 0:
        #print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)
        print(sess.run(cost, feed_dict={x1: x1_data,  Y: y_data}))

print(sess.run(w1))
print(sess.run(b))

