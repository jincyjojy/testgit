import tensorflow as tf  
import numpy as np
#defining training dataset
trainX = np.linspace(-1, 1, 101)
trainY = 3*trainX + np.random.randn(*trainX.shape)*0.33
print("hello1")
#defining placeholders
X = tf.placeholder("float")
Y = tf.placeholder("float")
#defining model
w = tf.Variable(0.0, name="weight")
y_model = tf.multiply(X, w)
cost = tf.pow((Y - y_model), 2)
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
print("hello2")
#initialize variables and defne session
init = tf.initialize_all_variables()
with tf.Session() as sess:
	sess.run(init)
	for i in range(100):
		for (x, y) in zip(trainX, trainY):
			sess.run(train_op, feed_dict={X:x, Y:y})
			print(sess.run(w))
print("hello3")
with tf.Session() as sess:
	sess.run(init)
	print(sess.run(w))
