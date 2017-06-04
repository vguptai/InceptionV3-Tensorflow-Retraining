import argparse
import sys
from constants import *
import tensorflow as tf
from PIL import Image
from osUtils import *
from collections import OrderedDict

parser = argparse.ArgumentParser()

parser.add_argument(
    '--num_top_predictions',
    type=int,
    default=1,
    help='Display this many predictions.')

parser.add_argument(
    '--path_to_graph',
    type=str,
    default='./tmp/output_graph.pb',
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

def load_graph(filename):
  """Unpersists graph from file as default graph."""
  with tf.gfile.FastGFile(filename, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

def load_labels(filename):
  """Read in labels, one label per line."""
  return [line.rstrip() for line in tf.gfile.GFile(filename)]

def run_graph(image_data, labels, input_layer_name, output_layer_name,
              num_top_predictions):
  with tf.Session() as sess:
    # Feed the image_data as input to the graph.
    #   predictions  will contain a two-dimensional array, where one
    #   dimension represents the input image count, and the other has
    #   predictions per class
    softmax_tensor = sess.graph.get_tensor_by_name(output_layer_name)
    predictions, = sess.run(softmax_tensor, {input_layer_name: image_data})
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

"Classify a file"
def label_a_file(file_path,labels,FLAGS):
    image_data = load_image("./cifar10Dataset/airplane/30.png")
    return run_graph(image_data, labels, DECODED_JPEG_DATA_TENSOR_NAME, FLAGS.output_layer,
              FLAGS.num_top_predictions)

"""
Simple test function to classify an image
"""
def test():
    file_path = "./cifar10Dataset/airplane/30.png"
    labels = load_labels(FLAGS.labels)
    load_graph(FLAGS.path_to_graph)
    label_a_file(file_path,labels,FLAGS)

"""
Classify the images present in the test folder and dump the results
in the file as per Kaggle submission format
"""
def kaggle_test():
    labels = load_labels(FLAGS.labels)
    load_graph(FLAGS.path_to_graph)
    print "Reading test images..."
    file_paths = get_images_from_directory("TestData")
    label_map = {}
    total_files = len(file_paths)
    print "Number of files to be classified..."
    file_index = 0
    for file_path in file_paths:
        file_index = file_index + 1
        print "Labelling files..."+str(file_index)+"/"+str(total_files)
        label = label_a_file(file_path,labels,FLAGS)
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
  #test()
  print(kaggle_test())
