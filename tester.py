import argparse
import sys
from constants import *
import tensorflow as tf
from PIL import Image
from osUtils import *
from collections import OrderedDict
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument(
    '--num_top_predictions',
    type=int,
    default=1,
    help='Display this many predictions.')

parser.add_argument(
    '--path_to_graph',
    type=str,
    default='./tmp/output_graph/1497085585/model_99.9977760712_92.0000025034.pb',
    #default='./tmp/output_graph/1496918099/model_99.8622210953_91.4600023031.pb',
    #default='./tmp/output_graph/1496818169/model_98.5133332544_91.0000015497.pb',
    #default='./tmp/output_graph/1496748387/model_94.6555585199_91.2000020742.pb',
    #default= './tmp/output_graph/1496559461/model_94.4044449329_90.8400014639.pb',
    help='Absolute path to graph file (.pb)')

parser.add_argument(
    '--labels',
    type=str,
    default='./tmp/output_labels.txt',
    help='Absolute path to labels file (.txt)')

parser.add_argument(
    '--output_layer',
    type=str,
    default='final_result:0',
    help='Name of the result operation')

def load_image(image_path):
    image = Image.open(image_path)
    image_data = image.convert('RGB')
    return image_data

def fix_graph(graph_def):
    # fix nodes
    for node in graph_def.node:
      if node.op == 'RefSwitch':
        node.op = 'Switch'
        for index in xrange(len(node.input)):
          if 'moving_' in node.input[index]:
            node.input[index] = node.input[index] + '/read'
      elif node.op == 'AssignSub':
        node.op = 'Sub'
        if 'use_locking' in node.attr: del node.attr['use_locking']

def load_graph(filename):
  """Unpersists graph from file as default graph."""
  print "Loading the graph..."
  with tf.gfile.FastGFile(filename, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    fix_graph(graph_def)
    tf.import_graph_def(graph_def, name='')
  for op in tf.get_default_graph().get_operations():
    print str(op.name)
    #print str(op.values)
  print "Graph loaded..."

def load_labels(filename):
  """Read in labels, one label per line."""
  return [line.rstrip() for line in tf.gfile.GFile(filename)]

def run_graph(sess, image_data, labels, input_layer_name, output_layer_name,
              num_top_predictions):

    # Feed the image_data as input to the graph.
    #   predictions  will contain a two-dimensional array, where one
    #   dimension represents the input image count, and the other has
    #   predictions per class
    softmax_tensor = sess.graph.get_tensor_by_name(output_layer_name)
    predictions, = sess.run(softmax_tensor, {input_layer_name: image_data, "input/dropout_keep_rate:0":1.0, "input/is_training_ph:0":False})
    # Sort to show labels in order of confidence
    # sort the prediction array along last axis (columns) in ascending order and then take
    # top K which would be at the last of this array as it is sorted in increasing order
    # [::-1] - read it in the reverse order
    top_k = predictions.argsort()[-num_top_predictions:][::-1]
    for node_id in top_k:
      human_string = labels[node_id]
      score = predictions[node_id]
      #print('%s (score = %.5f)' % (human_string, score))
    return human_string

"""
Classify a file
"""
def label_a_file(sess,file_path,labels,FLAGS):
    image_data = load_image(file_path)
    return run_graph(sess, image_data, labels, DECODED_JPEG_DATA_TENSOR_NAME, FLAGS.output_layer,
              FLAGS.num_top_predictions)


"""
Classify files by batching - Looks like it is not supported right now.
"""
def label_files(sess,file_paths,labels,FLAGS):
    image_datas = []
    for file_path in file_paths:
        image_datas.append(np.asarray(load_image(file_path)))
    image_datas = np.array(image_datas)
    return run_graph(sess, image_datas, labels, DECODED_JPEG_DATA_TENSOR_NAME, FLAGS.output_layer,
              FLAGS.num_top_predictions)

"""
Simple test function to classify a batch of images
"""
def test_batch():
    file_path1 = "./cifar10Dataset/airplane/30.png"
    file_path2 = "./cifar10Dataset/airplane/31.png"

    file_paths = []
    file_paths.append(file_path1)
    file_paths.append(file_path2)

    labels = load_labels(FLAGS.labels)
    load_graph(FLAGS.path_to_graph)
    with tf.Session() as sess:
        label_files(sess,file_paths,labels,FLAGS)


"""
Simple test function to classify an image
"""
def test():
    file_path = "./cifar10Dataset/airplane/30.png"
    labels = load_labels(FLAGS.labels)
    load_graph(FLAGS.path_to_graph)
    with tf.Session() as sess:
        return label_a_file(sess,file_path,labels,FLAGS)

"""
Classify the images present in the test folder and dump the results
in the file as per Kaggle submission format
"""
def kaggle_test():
    labels = load_labels(FLAGS.labels)
    load_graph(FLAGS.path_to_graph)
    print "Reading test images..."
    file_paths = get_images_from_directory("../Kaggle-CIFAR-Dataset/test")
    label_map = {}
    total_files = len(file_paths)
    with tf.Session() as sess:
        print "Number of files to be classified..."+str(total_files)
        file_index = 0
        for file_path in file_paths:
            file_index = file_index + 1
            print "Labelling files..."+str(file_index)+"/"+str(total_files)
            label = label_a_file(sess,file_path,labels,FLAGS)
            label_map[int(os.path.basename(file_path).split(".")[0])] = label
    print "Finished classifiying images..."
    label_map = OrderedDict(sorted(label_map.items(), key=lambda t: t[0]))
    print "Writing results in file..."
    createKaggleSubmissionFile(label_map)

"""
Dump the labels as per Kaggle Format
"""
def createKaggleSubmissionFile(label_map):
    f = open("./kaggle_result.txt","wb")
    f.write("id,label")
    f.write("\n")
    for key in label_map.keys():
        f.write(str(key)+","+label_map[key])
        f.write("\n")
    f.close()


if __name__ == '__main__':
  FLAGS, unparsed = parser.parse_known_args()
  #print(test())
  #test_batch()
  print(kaggle_test())
