import tensorflow as tf


#define variables
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)

#input and output
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
linear_model = W*x + b


#loss
loss = tf.reduce_sum(tf.square(linear_model - y))  # reduce_sum不明
#optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)  # 学习效率
train = optimizer.minimize(loss)

# training data
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for i in range(100):
    sess.run(train, feed_dict={x: x_train, y: y_train})
    curr_W = sess.run([W])
    print(curr_W)





