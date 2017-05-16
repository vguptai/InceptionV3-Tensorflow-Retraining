import tensorflow as tf
from constants import *
import numpy as np

class InceptionV3:

	bottleneckTensor = None
	finalTensor = None
	groundTruthInput = None
	trainStep = None
	evaluationStep = None
	bottleneckInput = None
	inceptionGraph = None
	jpeg_data_tensor = None

	def __init__(self,modelPath):
		self._create_inception_graph(modelPath)

	def _create_inception_graph(self,modelPath):
  		with tf.Session() as sess:
    			with tf.gfile.FastGFile(modelPath, 'rb') as f:
      				graph_def = tf.GraphDef()
      				graph_def.ParseFromString(f.read())
      				self.bottleneckTensor, self.jpeg_data_tensor, resized_input_tensor = (tf.import_graph_def(graph_def, name='', return_elements=[BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME,RESIZED_INPUT_TENSOR_NAME]))

	def add_final_training_ops(self,class_count, final_tensor_name, learningRate):
		with tf.name_scope('input'):
		    self.bottleneckInput = tf.placeholder_with_default(self.bottleneckTensor, shape=[None, BOTTLENECK_TENSOR_SIZE],name='BottleneckInputPlaceholder')
		    self.groundTruthInput = tf.placeholder(tf.float32,[None, class_count],name='GroundTruthInput')

		layer_name = 'final_training_ops'
		with tf.name_scope(layer_name):
		    with tf.name_scope('weights'):
		    	initial_value = tf.truncated_normal([BOTTLENECK_TENSOR_SIZE, class_count],stddev=0.001)
		      	layer_weights = tf.Variable(initial_value, name='final_weights')
		    with tf.name_scope('biases'):
		      	layer_biases = tf.Variable(tf.zeros([class_count]), name='final_biases')
		    with tf.name_scope('Wx_plus_b'):
		      	logits = tf.matmul(self.bottleneckInput, layer_weights) + layer_biases

		self.finalTensor = tf.nn.softmax(logits, name=final_tensor_name)
		with tf.name_scope('cross_entropy'):
			self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.groundTruthInput, logits=logits)
		    	with tf.name_scope('total'):
		    		self.cross_entropy_mean = tf.reduce_mean(self.cross_entropy)

		with tf.name_scope('train'):
			optimizer = tf.train.GradientDescentOptimizer(learningRate)
		    	self.trainStep = optimizer.minimize(self.cross_entropy_mean)

	def add_evaluation_step(self):
		with tf.name_scope('accuracy'):
			with tf.name_scope('correct_prediction'):
				prediction = tf.argmax(self.finalTensor, 1)
				correctPrediction = tf.equal(prediction, tf.argmax(self.groundTruthInput, 1))
		with tf.name_scope('accuracy'):
			self.evaluationStep = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))
		return self.evaluationStep, prediction

	def train_step(self,sess,train_bottlenecks,train_ground_truth):
		sess.run([self.trainStep],feed_dict={self.bottleneckInput: train_bottlenecks,self.groundTruthInput: train_ground_truth})

	def evaluate(self,sess,data_bottlenecks,data_ground_truth):
		accuracy, crossEntropyValue = sess.run([self.evaluationStep, self.cross_entropy_mean],feed_dict={self.bottleneckInput: data_bottlenecks,self.groundTruthInput: data_ground_truth})
		return accuracy,crossEntropyValue

	def run_bottleneck_on_image(self,sess, image_data):
	  bottleneck_values = sess.run(self.bottleneckTensor,{self.jpeg_data_tensor: image_data})
	  bottleneck_values = np.squeeze(bottleneck_values)
	  return bottleneck_values
