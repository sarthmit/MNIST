import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def initializer(shape):
	initial = tf.truncated_normal(shape, stddev=0.075)
	return tf.Variable(initial)

def regularization(data):
	return tf.reduce_mean(tf.square(data))

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

x = tf.placeholder(tf.float32, [None,784])
y = tf.placeholder(tf.float32, [None,10])

il = tf.constant(784)
hl1 = tf.constant(2500)
hl2 = tf.constant(2000)
hl3 = tf.constant(1500)
hl4 = tf.constant(1000)
hl5 = tf.constant(500)
ol = tf.constant(10)
reg = tf.constant(2.5)

W1 = initializer([il,hl1])
b1 = initializer([hl1])

W2 = initializer([hl1,hl2])
b2 = initializer([hl2])

W3 = initializer([hl2,hl3])
b3 = initializer([hl3])

W4 = initializer([hl3,hl4])
b4 = initializer([hl4])

W5 = initializer([hl4,hl5])
b5 = initializer([hl5])

W_out = initializer([hl5,ol])
b_out = initializer([ol])

ff1 = tf.nn.sigmoid(tf.matmul(x,W1)+b1)

ff2 = tf.nn.sigmoid(tf.matmul(ff1,W2)+b2)

ff3 = tf.nn.sigmoid(tf.matmul(ff2,W3)+b3)

ff4 = tf.nn.sigmoid(tf.matmul(ff3,W4)+b4)

ff5 = tf.nn.sigmoid(tf.matmul(ff4,W5)+b5)

ff_out = tf.nn.sigmoid(tf.matmul(ff5,W_out)+b_out)

regul = regularization(W1)+regularization(W2)+regularization(W3) + regularization(W4) + regularization(W5) + regularization(W_out)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=ff_out)) + reg*regul/6
optimizer = tf.train.AdamOptimizer().minimize(cost)

correct = tf.equal(tf.argmax(ff_out,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))

init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)

for i in range(10000):
	batch = mnist.train.next_batch(200)
	if(i%100 == 0):
		train_accuracy = accuracy.eval(feed_dict={x:batch[0], y:batch[1]})
		print("Step %d   Training Accuracy: %f" %((i/100 + 1), train_accuracy))

	optimizer.run(feed_dict={x:batch[0], y:batch[1]})

print("Test Accuracy: %f" %accuracy.eval(feed_dict={x:mnist.test.images, y:mnist.test.labels}))