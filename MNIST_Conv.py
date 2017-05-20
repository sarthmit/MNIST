import tensorflow as tf  
from tensorflow.examples.tutorials.mnist import input_data

def initializer(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def conv2d(x,W):
	return tf.nn.conv2d(x , W , [1,1,1,1] , padding="SAME")

def max_pool(x):
	return tf.nn.max_pool(x , [1,2,2,1] , [1,2,2,1] , padding="SAME")

mnist = input_data.read_data_sets("MNIST_data", one_hot = True)

x = tf.placeholder(tf.float32, [None,784])
y = tf.placeholder(tf.float32, [None,10])
keep_prob = tf.placeholder(tf.float32)

W_conv1 = initializer([5 , 5 , 1 , 48])
b_conv1 = initializer([48])

W_conv2 = initializer([5 , 5 , 48 , 64])
b_conv2 = initializer([64])

W_fc1 = initializer([7*7*64 , 1024])
b_fc1 = initializer([1024])

W_fc2 = initializer([1024 , 1024])
b_fc2 = initializer([1024])

W_out = initializer([1024 , 10])
b_out = initializer([10])

x_image = tf.reshape(x , [-1,28,28,1])

h1 = tf.nn.relu(conv2d(x_image , W_conv1) + b_conv1)
h1 = max_pool(h1)

h2 = tf.nn.relu(conv2d(h1 , W_conv2) + b_conv2)
h2 = max_pool(h2)

h2 = tf.reshape(h2 , [-1,7*7*64])

h3 = tf.nn.relu(tf.matmul(h2 , W_fc1) + b_fc1)

h3 = tf.nn.dropout(h3 , keep_prob)

h4 = tf.nn.sigmoid(tf.matmul(h3 , W_fc2) + b_fc2)

h4 = tf.nn.dropout(h4 , keep_prob)

h_out = tf.nn.sigmoid(tf.matmul(h4 , W_out) + b_out)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=h_out))

train = tf.train.AdamOptimizer().minimize(cost)

correct_pred = tf.equal(tf.argmax(h_out, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)

for i in range(5000):
	batch = mnist.train.next_batch(250)
	if(i%100 == 0):
		train_accuracy = accuracy.eval(feed_dict={x:batch[0], y:batch[1], keep_prob:1.0})
		print("Step %d   Training Accuracy: %f" %((i/100 + 1), train_accuracy))
	train.run(feed_dict={x:batch[0], y:batch[1], keep_prob:0.5})

sum=0.0
for i in range(10):
	batch_x = mnist.test.images[(i*1000):((i+1)*1000)-1]
	batch_y = mnist.test.labels[(i*1000):((i+1)*1000)-1]
	sum = sum + accuracy.eval(feed_dict={x:batch_x, y:batch_y, keep_prob:1.0})
print("Test Accuracy: %f" %(sum/10.0))