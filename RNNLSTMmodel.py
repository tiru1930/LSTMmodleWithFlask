import tensorflow as tf
import numpy as numpy
from tensorflow.contrib import rnn
from create_sentiment_featuresets import create_feature_sets_and_labels
train_x,train_y,test_x,test_y = create_feature_sets_and_labels('pos.txt','neg.txt')

print(test_x[0])
print("TEST LABEL")
print(test_y[0])
learning_rate=0.001
batch_size=128
display_step=10
epchos=1000

n_input=423
n_steps=1
n_hidden=128
n_outputs=2

x=tf.placeholder("float",[None,n_steps,n_input])
y=tf.placeholder("float",[None,n_outputs])

weights=tf.Variable(tf.random_normal([n_hidden,n_outputs]),name="weights")
biases=tf.Variable(tf.random_normal([n_outputs]),name="biases")
		


def RNN(x,weights,biases):
	x=tf.transpose(x,[1,0,2])
	x=tf.reshape(x,[-1,n_input])
	x=tf.split(x,n_steps,0)
	lstm_cell=rnn.BasicLSTMCell(n_hidden,forget_bias=1.0)
	outputs,states=rnn.static_rnn(lstm_cell,x,dtype=tf.float32)
	return tf.matmul(outputs[-1],weights)+biases

pred=RNN(x,weights,biases)
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
correct_pred=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))
init=tf.global_variables_initializer()
saver = tf.train.Saver(tf.trainable_variables())

with tf.Session() as sess:
	sess.run(init)
	step=0
	while step<epchos:
		start=step
		end=step+batch_size
		batch_x=numpy.array(train_x[step:end])
		batch_y=numpy.array(train_y[step:end])
		batch_x=batch_x.reshape(batch_size,n_steps,n_input)
		sess.run(optimizer,feed_dict={x:batch_x,y:batch_y})
		acc=sess.run(accuracy,feed_dict={x:batch_x,y:batch_y})
		loss=sess.run(cost,feed_dict={x:batch_x,y:batch_y})
		#save_path=saver.save(sess,"data/RNNModel.ckpt",step,write_meta_graph=False, write_state=False)
		print("Accuracy" + "{:.6f}".format(acc) + "loos" + "{:,.5f}".format(loss))
		step+=1

	save_path=saver.save(sess,"data/RNNModel.ckpt",write_meta_graph=False, write_state=False)
	print("Model Saved in file" +  save_path)
	test_data=numpy.array(test_x).reshape((-1, n_steps, n_input))
	test_label=test_y
	print ("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label}))
