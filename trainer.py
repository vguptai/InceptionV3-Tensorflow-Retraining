import argparse
import preprocessor as preprocessor
from constants import *
from InceptionV3 import *
import os.path
from datetime import datetime
from DatasetBatcher import *
import DatasetManager as DatasetManager
from tensorflow.python.framework import graph_util
import time
from osUtils import *

def create_inception_graph(num_classes,num_batches_per_epoch,FLAGS):
    modelFilePath = os.path.join(FLAGS.imagenet_inception_model_dir, INCEPTION_MODEL_GRAPH_DEF_FILE)
    inceptionV3 = InceptionV3(modelFilePath)
    inceptionV3.add_final_training_ops(num_classes,FLAGS.final_tensor_name,FLAGS.optimizer_name,num_batches_per_epoch, FLAGS)
    inceptionV3.add_evaluation_step()
    return inceptionV3

def setup_image_distortion_ops(inceptionV3,FLAGS):
    if DatasetManager.should_distort_images(FLAGS):
        inceptionV3.add_input_distortions(FLAGS.flip_left_right, FLAGS.random_crop,FLAGS.random_scale, FLAGS.random_brightness)

def train_an_epoch(sess,inceptionV3,datasetBatcher,FLAGS):
    datasetBatcher.reset_training_offset(FLAGS.shuffle_dataset_every_epoch)
    train_image_paths,train_ground_truth,train_labels = datasetBatcher.get_next_training_batch(FLAGS.train_batch_size)
    while train_image_paths is not None:
        if DatasetManager.should_distort_images(FLAGS):
            train_bottlenecks = DatasetManager.get_random_distorted_bottlenecks(sess,train_image_paths,inceptionV3)
        else:
            train_bottlenecks = DatasetManager.get_random_cached_bottlenecks_new(sess,train_image_paths,train_labels,FLAGS.bottleneck_dir,inceptionV3)
        inceptionV3.train_step(sess,train_bottlenecks,train_ground_truth,FLAGS.dropout_keep_rate)
        train_image_paths,train_ground_truth,train_labels = datasetBatcher.get_next_training_batch(FLAGS.train_batch_size)
    datasetBatcher.reset_training_offset()

def get_next_batch(phase,datasetBatcher):
    if phase == "training":
        image_paths,ground_truth,labels = datasetBatcher.get_next_training_batch(FLAGS.train_batch_size)
    elif phase == "testing":
        image_paths,ground_truth,labels = datasetBatcher.get_next_testing_batch(FLAGS.test_batch_size)
    elif phase == "validation":
        image_paths,ground_truth,labels = datasetBatcher.get_next_validation_batch(FLAGS.validation_batch_size)
    return image_paths,ground_truth,labels

def evaluate_accuracy(sess,phase,inceptionV3,datasetBatcher,FLAGS,epoch_index):
    if phase == "training":
        datasetBatcher.reset_training_offset()
    elif phase == "testing":
        datasetBatcher.reset_testing_offset()
    elif phase == "validation":
        datasetBatcher.reset_validation_offset()

    image_paths,ground_truth,labels = get_next_batch(phase,datasetBatcher)
    batch_index = 0
    accuracy = 0
    cross_entropy_value = 0
    num_samples = 0
    while image_paths is not None:
        batch_index = batch_index + 1
        bottlenecks = DatasetManager.get_random_cached_bottlenecks_new(sess,image_paths,labels,FLAGS.bottleneck_dir,inceptionV3)
        accuracy_batch, cross_entropy_value_batch = inceptionV3.evaluate(sess,bottlenecks,ground_truth)
        num_samples = num_samples + len(bottlenecks)
        accuracy = accuracy + accuracy_batch
        cross_entropy_value = cross_entropy_value + cross_entropy_value_batch
        image_paths,ground_truth,labels = get_next_batch(phase,datasetBatcher)

    if batch_index == 0:
        print "No samples to evaluate in this phase:" + phase
    else:
        accuracy = accuracy * 100/batch_index
        print('%s: Step %d: %s Accuracy = %.1f%%' % (datetime.now(), epoch_index,phase,accuracy))
        print('%s: Step %d: %s Cross entropy = %f' % (datetime.now(), epoch_index,phase,cross_entropy_value/batch_index))
    return accuracy,cross_entropy_value

def train_graph(inceptionV3,datasetBatcher,FLAGS):
    with tf.Session(graph=inceptionV3.inceptionGraph) as sess:
        start_time = str(int(time.time()))
        init = tf.global_variables_initializer()
        sess.run(init)
        print "Training the model with dropout rate:" + str(FLAGS.dropout_keep_rate)
        for i in range(FLAGS.how_many_training_steps):
            print "Epoch..."+str(i)+"/"+str(FLAGS.how_many_training_steps)
            train_an_epoch(sess,inceptionV3,datasetBatcher,FLAGS)
            is_last_step = (i + 1 == FLAGS.how_many_training_steps)
            if (i % FLAGS.eval_step_interval) == 0 or is_last_step:
                train_accuracy,_ = evaluate_accuracy(sess,"training",inceptionV3,datasetBatcher,FLAGS,i)
                validation_accuracy,_ = evaluate_accuracy(sess,"validation",inceptionV3,datasetBatcher,FLAGS,i)
                model_name = "model_"+str(train_accuracy)+"_"+str(validation_accuracy)+".pb"
                save_graph(sess,inceptionV3.inceptionGraph,start_time,model_name,FLAGS)
        evaluate_accuracy(sess,"testing",inceptionV3,datasetBatcher,FLAGS,i)
        save_graph(sess,inceptionV3.inceptionGraph,start_time,"model_final.pb",FLAGS)

def save_graph(sess,graph,prefix,model_name,FLAGS):
    output_graph_def = graph_util.convert_variables_to_constants(
        sess, graph.as_graph_def(), [FLAGS.final_tensor_name])
    sub_dir_path = os.path.join(FLAGS.output_graph, prefix)
    ensure_dir_exists(sub_dir_path)
    output_graph_path = os.path.join(sub_dir_path,model_name)
    print "Saving the graph at:"+ output_graph_path
    with tf.gfile.FastGFile(output_graph_path, 'wb') as f:
      f.write(output_graph_def.SerializeToString())

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--image_dir',
      type=str,
      default='cifar10Dataset',
      help='Path to folders of labeled images.'
  )
  parser.add_argument(
      '--output_graph',
      type=str,
      default='./tmp/output_graph',
      help='Where to save the trained graph.'
  )
  parser.add_argument(
      '--output_labels',
      type=str,
      default='./tmp/output_labels.txt',
      help='Where to save the trained graph\'s labels.'
  )
  parser.add_argument(
      '--summaries_dir',
      type=str,
      default='./tmp/retrain_logs',
      help='Where to save summary logs for TensorBoard.'
  )
  parser.add_argument(
      '--how_many_training_steps',
      type=int,
      default=500,
      help='How many training steps to run before ending.'
  )
  parser.add_argument(
      '--optimizer_name',
      type=str,
      default="sgd",
      help='Optimizer to be used: sgd,adam,rmsprop'
  )
  parser.add_argument(
      '--learning_rate_decay_factor',
      type=float,
      default="0.16",
      help='Learning rate decay factor.'
  )
  parser.add_argument(
      '--use_batch_normalization',
      type=bool,
      default=True,
      help='Control the use of batch normalization'
  )
  parser.add_argument(
      '--learning_rate',
      type=float,
      default=0.1,
      help='Initial learning rate.'
  )
  parser.add_argument(
      '--rmsprop_decay',
      type=float,
      default=0.9,
      help='Decay term for RMSProp.'
  )
  parser.add_argument(
      '--rmsprop_momentum',
      type=float,
      default=0.9,
      help='Momentum in RMSProp.'
  )
  parser.add_argument(
      '--rmsprop_epsilon',
      type=float,
      default=1.0,
      help='Epsilon term for RMSProp.'
  )
  parser.add_argument(
      '--num_epochs_per_decay',
      type=int,
      default=30,
      help='Epochs after which learning rate decays.'
  )
  parser.add_argument(
      '--learning_rate_type',
      type=str,
      default="exp_decay",
      help='exp_decay,const'
  )
  parser.add_argument(
      '--testing_percentage',
      type=int,
      default=0,
      help='What percentage of images to use as a test set.'
  )
  parser.add_argument(
      '--validation_percentage',
      type=int,
      default=10,
      help='What percentage of images to use as a validation set.'
  )
  parser.add_argument(
      '--eval_step_interval',
      type=int,
      default=10,
      help='How often to evaluate the training results.'
  )
  parser.add_argument(
      '--train_batch_size',
      type=int,
      default=100,
      help='How many images to train on at a time.'
  )
  parser.add_argument(
      '--test_batch_size',
      type=int,
      default=100,
      help="""\
      How many images to test on. This test set is only used once, to evaluate
      the final accuracy of the model after training completes.
      A value of -1 causes the entire test set to be used, which leads to more
      stable results across runs.\
      """
  )
  parser.add_argument(
      '--validation_batch_size',
      type=int,
      default=100,
      help="""\
      How many images to use in an evaluation batch. This validation set is
      used much more often than the test set, and is an early indicator of how
      accurate the model is during training.
      A value of -1 causes the entire validation set to be used, which leads to
      more stable results across training iterations, but may be slower on large
      training sets.\
      """
  )
  parser.add_argument(
      '--print_misclassified_test_images',
      default=False,
      help="""\
      Whether to print out a list of all misclassified test images.\
      """,
      action='store_true'
  )
  parser.add_argument(
      '--imagenet_inception_model_dir',
      type=str,
      default='./imagenetInception',
      help="""\
      Path to classify_image_graph_def.pb,
      imagenet_synset_to_human_label_map.txt, and
      imagenet_2012_challenge_label_map_proto.pbtxt.\
      """
  )
  parser.add_argument(
      '--bottleneck_dir',
      type=str,
      default='./tmp/bottleneck',
      help='Path to cache bottleneck layer values as files.'
  )
  parser.add_argument(
      '--final_tensor_name',
      type=str,
      default='final_result',
      help="""\
      The name of the output classification layer in the retrained graph.\
      """
  )
  parser.add_argument(
      '--dropout_keep_rate',
      type=float,
      default=0.5,
      help="""\
      Dropout rate used while training
      """
    )
  parser.add_argument(
      '--apply_distortions',
      default=False,
      help="""\
      Apply distortions to images while training.\
      """
  )
  parser.add_argument(
      '--shuffle_dataset_every_epoch',
      default=True,
      help="""\
      Shuffle the training dataset at every epoch.\
      """
  )
  parser.add_argument(
      '--flip_left_right',
      default=True,
      help="""\
      Whether to randomly flip half of the training images horizontally.\
      """,
      action='store_true'
  )
  parser.add_argument(
      '--random_crop',
      type=int,
      default=20,
      help="""\
      A percentage determining how much of a margin to randomly crop off the
      training images.\
      """
  )
  parser.add_argument(
      '--random_scale',
      type=int,
      default=20,
      help="""\
      A percentage determining how much to randomly scale up the size of the
      training images by.\
      """
  )
  parser.add_argument(
      '--random_brightness',
      type=int,
      default=20,
      help="""\
      A percentage determining how much to randomly multiply the training image
      input pixels up or down by.\
      """
  )

  FLAGS, unparsed = parser.parse_known_args()
  preprocessor.setup(FLAGS)

  imageMap = DatasetManager.readDataset(FLAGS)
  numClasses = len(imageMap.keys())

  datasetBatcher = DatasetBatcher(imageMap,FLAGS.image_dir)
  num_training_batches = datasetBatcher.number_of_training_batches(FLAGS.train_batch_size)

  inceptionV3 = create_inception_graph(numClasses,num_training_batches, FLAGS)
  setup_image_distortion_ops(inceptionV3,FLAGS)

  train_graph(inceptionV3,datasetBatcher,FLAGS)
