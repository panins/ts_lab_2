import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

iris = datasets.load_iris()
x_vals = np.array([x[0:3] for x in iris.data])
y_vals = np.array([x[3] for x in iris.data])
# print(y_vals)

sess = tf.Session()

seed = 2
tf.set_random_seed(seed)
np.random.seed(seed)


train_indices  = np.random.choice(len(x_vals),round(len(x_vals)*0.8),replace=False) # replace 同一个元素只能选一次
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))

x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]
def normalize_cols(m):
    col_max = m.max(axis=0) # 纵轴的最大值
    col_min = m.min(axis=0)
    return (m-col_min)/(col_max-col_min)

x_vals_train = np.nan_to_num(normalize_cols(x_vals_train))
x_vals_test = np.nan_to_num(normalize_cols(x_vals_test))

batch_size = 50
x_data = tf.placeholder(shape=[None,3],dtype=tf.float32)
y_target= tf.placeholder(shape=[None,1],dtype=tf.float32)

# 声明合适形状的变量
hidden_layer_nodes = 100
A1 = tf.Variable(tf.random_normal(shape=[3,hidden_layer_nodes]))
b1 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes]))
A2 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes,1]))
b2 = tf.Variable(tf.random_normal(shape=[1]))



hidden_output = tf.nn.relu(tf.add(tf.matmul(x_data,A1),b1))
final_output = tf.nn.relu(tf.add(tf.matmul(hidden_output,A2),b2))

loss = tf.reduce_mean(tf.square(y_target - final_output))

my_opt = tf.train.GradientDescentOptimizer(0.005)
train_step = my_opt.minimize(loss)
init = tf.global_variables_initializer()
sess.run(init)

loss_vec = []
test_loss = []
for i in range(50000):
    rand_index = np.random.choice(len(x_vals_train), size=batch_size)
    rand_x = x_vals_train[rand_index]
    rand_y = np.transpose([y_vals_train[rand_index]])

    sess.run(train_step, feed_dict={x_data:rand_x, y_target:rand_y})
    temp_loss = sess.run(loss, feed_dict={x_data:rand_x, y_target:rand_y})
    loss_vec.append(np.sqrt(temp_loss))
    test_temp_loss = sess.run(loss, feed_dict={x_data:x_vals_test, y_target:np.transpose([y_vals_test])})
    test_loss.append(np.sqrt(test_temp_loss))
    # if (i+1)%50 == 0:
    #     print(rand_index)

plt.plot(loss_vec, 'k-',label='Train loss')
plt.plot(test_loss, 'r--',label='test loss')
plt.legend(loc='upper right')
plt.show()