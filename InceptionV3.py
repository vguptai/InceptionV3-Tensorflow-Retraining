import tensorflow as tf
from constants import *
import numpy as np
from tensorflow.python.framework import tensor_shape

class InceptionV3:

	bottleneckTensor = None
	finalTensor = None
	groundTruthInput = None
	trainStep = None
	evaluationStep = None
	bottleneckInput = None
	inceptionGraph = None
	jpeg_data_tensor = None
	distortion_image_data_input_placeholder = None
	distort_image_data_operation = None

	def __init__(self,modelPath):
		self._create_inception_graph(modelPath)

	def _create_inception_graph(self,modelPath):
		with tf.Graph().as_default() as self.inceptionGraph:
			with tf.gfile.FastGFile(modelPath, 'rb') as f:
					graph_def = tf.GraphDef()
					graph_def.ParseFromString(f.read())
					self.bottleneckTensor, self.jpeg_data_tensor, resized_input_tensor, self.decoded_jpeg_data_tensor = (tf.import_graph_def(graph_def, name='', return_elements=[BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME,RESIZED_INPUT_TENSOR_NAME,DECODED_JPEG_DATA_TENSOR_NAME]))

	def add_final_training_ops(self,class_count, final_tensor_name, learningRate):
		with self.inceptionGraph.as_default():
			with tf.name_scope('input'):
			    self.bottleneckInput = tf.placeholder_with_default(self.bottleneckTensor, shape=[None, BOTTLENECK_TENSOR_SIZE],name='BottleneckInputPlaceholder')
			    self.groundTruthInput = tf.placeholder(tf.float32,[None, class_count],name='GroundTruthInput')


			layer_name = 'final_minus_1_training_ops'
			with tf.name_scope(layer_name):
				with tf.name_scope('weights'):
					initial_value_final_minus_1 = tf.truncated_normal([BOTTLENECK_TENSOR_SIZE, FINAL_MINUS_1_LAYER_SIZE],stddev=0.001)
					layer_weights_final_minus_1 = tf.Variable(initial_value_final_minus_1, name='final_weights')
				with tf.name_scope('biases'):
					layer_biases_final_minus_1 = tf.Variable(tf.zeros([FINAL_MINUS_1_LAYER_SIZE]), name='final_biases')
				with tf.name_scope('Wx_plus_b'):
					logits_final_minus_1 = tf.matmul(self.bottleneckInput, layer_weights_final_minus_1) + layer_biases_final_minus_1

			layer_name = 'final_training_ops'
			with tf.name_scope(layer_name):
			    with tf.name_scope('weights'):
			    	initial_value = tf.truncated_normal([FINAL_MINUS_1_LAYER_SIZE, class_count],stddev=0.001)
			      	layer_weights = tf.Variable(initial_value, name='final_weights')
			    with tf.name_scope('biases'):
			      	layer_biases = tf.Variable(tf.zeros([class_count]), name='final_biases')
			    with tf.name_scope('Wx_plus_b'):
			      	logits = tf.matmul(logits_final_minus_1, layer_weights) + layer_biases

			self.finalTensor = tf.nn.softmax(logits, name=final_tensor_name)
			with tf.name_scope('cross_entropy'):
				self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.groundTruthInput, logits=logits)
			    	with tf.name_scope('total'):
			    		self.cross_entropy_mean = tf.reduce_mean(self.cross_entropy)

			with tf.name_scope('train'):
				optimizer = tf.train.GradientDescentOptimizer(learningRate)
			    	self.trainStep = optimizer.minimize(self.cross_entropy_mean)

	def add_evaluation_step(self):
		with self.inceptionGraph.as_default():
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
	  #bottleneck_values = sess.run(self.bottleneckTensor,{self.jpeg_data_tensor: image_data})
	  bottleneck_values = sess.run(self.bottleneckTensor,{self.decoded_jpeg_data_tensor: image_data})
	  bottleneck_values = np.squeeze(bottleneck_values)
	  return bottleneck_values

	def distort_image(self,sess,image_data):
		return sess.run(self.distort_image_data_operation ,{self.distortion_image_data_input_placeholder: image_data})

	def add_input_distortions(self, flip_left_right, random_crop, random_scale,
	                          random_brightness):
	  """Creates the operations to apply the specified distortions.
	  During training it can help to improve the results if we run the images
	  through simple distortions like crops, scales, and flips. These reflect the
	  kind of variations we expect in the real world, and so can help train the
	  model to cope with natural data more effectively. Here we take the supplied
	  parameters and construct a network of operations to apply them to an image.
	  Cropping
	  ~~~~~~~~
	  Cropping is done by placing a bounding box at a random position in the full
	  image. The cropping parameter controls the size of that box relative to the
	  input image. If it's zero, then the box is the same size as the input and no
	  cropping is performed. If the value is 50%, then the crop box will be half the
	  width and height of the input. In a diagram it looks like this:
	  <       width         >
	  +---------------------+
	  |                     |
	  |   width - crop%     |
	  |    <      >         |
	  |    +------+         |
	  |    |      |         |
	  |    |      |         |
	  |    |      |         |
	  |    +------+         |
	  |                     |
	  |                     |
	  +---------------------+
	  Scaling
	  ~~~~~~~
	  Scaling is a lot like cropping, except that the bounding box is always
	  centered and its size varies randomly within the given range. For example if
	  the scale percentage is zero, then the bounding box is the same size as the
	  input and no scaling is applied. If it's 50%, then the bounding box will be in
	  a random range between half the width and height and full size.
	  Args:
	    flip_left_right: Boolean whether to randomly mirror images horizontally.
	    random_crop: Integer percentage setting the total margin used around the
	    crop box.
	    random_scale: Integer percentage of how much to vary the scale by.
	    random_brightness: Integer range to randomly multiply the pixel values by.
	    graph.
	  Returns:
	    The jpeg input layer and the distorted result tensor.
	  """
	  print "Setting up image distortion operations..."
	  #jpeg_data = tf.placeholder(tf.string, name='DistortJPGInput')
	  #decoded_image = tf.image.decode_jpeg(jpeg_data, channels=MODEL_INPUT_DEPTH)
	  with self.inceptionGraph.as_default():
		  decoded_image_as_float = tf.placeholder('float', [None,None,MODEL_INPUT_DEPTH])
		  decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
		  margin_scale = 1.0 + (random_crop / 100.0)
		  resize_scale = 1.0 + (random_scale / 100.0)
		  margin_scale_value = tf.constant(margin_scale)
		  resize_scale_value = tf.random_uniform(tensor_shape.scalar(),
		                                         minval=1.0,
		                                         maxval=resize_scale)
		  scale_value = tf.multiply(margin_scale_value, resize_scale_value)
		  precrop_width = tf.multiply(scale_value, MODEL_INPUT_WIDTH)
		  precrop_height = tf.multiply(scale_value, MODEL_INPUT_HEIGHT)
		  precrop_shape = tf.stack([precrop_height, precrop_width])
		  precrop_shape_as_int = tf.cast(precrop_shape, dtype=tf.int32)
		  precropped_image = tf.image.resize_bilinear(decoded_image_4d,
		                                              precrop_shape_as_int)
		  precropped_image_3d = tf.squeeze(precropped_image, squeeze_dims=[0])
		  cropped_image = tf.random_crop(precropped_image_3d,
		                                 [MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH,
		                                  MODEL_INPUT_DEPTH])
		  if flip_left_right:
		    flipped_image = tf.image.random_flip_left_right(cropped_image)
		  else:
		    flipped_image = cropped_image
		  brightness_min = 1.0 - (random_brightness / 100.0)
		  brightness_max = 1.0 + (random_brightness / 100.0)
		  brightness_value = tf.random_uniform(tensor_shape.scalar(),
		                                       minval=brightness_min,
		                                       maxval=brightness_max)
		  distort_result = tf.multiply(flipped_image, brightness_value)
		  #distort_result = tf.expand_dims(brightened_image, 0, name='DistortResult')

		  self.distortion_image_data_input_placeholder = decoded_image_as_float
		  self.distort_image_data_operation = distort_result
