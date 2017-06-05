import tensorflow as tf
import tarfile
from osUtils import *
import random
from constants import *
from six.moves import urllib
import sys
import numpy as np
from PIL import Image

def save_labels(FLAGS,image_map):
    print "Saving labels at:"+FLAGS.output_labels
    with tf.gfile.FastGFile(FLAGS.output_labels, 'w') as f:
      f.write('\n'.join(image_map.keys()) + '\n')

def splitInTrainTestValSet(fileList,testPercentage,validationPercentage):
  totalFiles = len(fileList)
  random.shuffle(fileList)
  trainPercentage = 100-testPercentage-validationPercentage
  numTrainingFiles = int(totalFiles*trainPercentage/100)
  numTestingFiles = int(totalFiles*testPercentage/100)
  numValidationFiles = int(totalFiles*validationPercentage/100)

  trainingSet = fileList[:numTrainingFiles]
  testingSet = fileList[numTrainingFiles:numTrainingFiles+numTestingFiles]
  validationSet = fileList[numTrainingFiles+numTestingFiles:]

  return trainingSet,testingSet,validationSet

def create_image_lists(image_dir, testingPercentage, validationPercentage):
  if not tf.gfile.Exists(image_dir):
    print("Image directory '" + image_dir + "' not found.")
    return None

  result = {}
  # Get all the subdirectories. Each subdirectory would correspond to a class
  subDirs = getSubdirectories(image_dir)
  print "Processing Images from the subdirectories..."+str(subDirs)

  # Valid image extensions
  extensions = ['jpg', 'jpeg', 'JPG', 'JPEG', 'png', 'PNG']
  fileListMap=getFilesFromDirectoriesWithExtension(subDirs,extensions)
  for key,value in fileListMap.iteritems():
	print "Number of Images in class:"+key+":"+str(len(value))
	trainingSet,testingSet,validationSet = splitInTrainTestValSet(value,testingPercentage,validationPercentage)
    	result[key] = {
        	'dir': key,
        	'training': trainingSet,
        	'testing': testingSet,
        	'validation': validationSet,
    	}
  return result

def get_bottleneck_path_new(image_path, label_name, bottleneck_dir):
	image_path = os.path.join(bottleneck_dir, label_name, os.path.basename(image_path))
	return image_path + '.txt'

def get_bottleneck_path(image_lists, label_name, index, bottleneck_dir,
                        category):
  """"Returns a path to a bottleneck file for a label at the given index.
  Args:
    image_lists: Dictionary of training images for each label.
    label_name: Label string we want to get an image for.
    index: Integer offset of the image we want. This will be moduloed by the
    available number of images for the label, so it can be arbitrarily large.
    bottleneck_dir: Folder string holding cached files of bottleneck values.
    category: Name string of set to pull images from - training, testing, or
    validation.
  Returns:
    File system path string to an image that meets the requested parameters.
  """
  return get_image_path(image_lists, label_name, index, bottleneck_dir,
                        category) + '.txt'

def get_image_path(image_lists, label_name, index, image_dir, category):
  """"Returns a path to an image for a label at the given index.
  Args:
    image_lists: Dictionary of training images for each label.
    label_name: Label string we want to get an image for.
    index: Int offset of the image we want. This will be moduloed by the
    available number of images for the label, so it can be arbitrarily large.
    image_dir: Root folder string of the subfolders containing the training
    images.
    category: Name string of set to pull images from - training, testing, or
    validation.
  Returns:
    File system path string to an image that meets the requested parameters.
  """
  if label_name not in image_lists:
    tf.logging.fatal('Label does not exist %s.', label_name)
  label_lists = image_lists[label_name]
  if category not in label_lists:
    tf.logging.fatal('Category does not exist %s.', category)
  category_list = label_lists[category]
  if not category_list:
    tf.logging.fatal('Label %s has no images in the category %s.',
                     label_name, category)
  mod_index = index % len(category_list)
  base_name = category_list[mod_index]
  sub_dir = label_lists['dir']
  full_path = os.path.join(image_dir, sub_dir, base_name)
  return full_path

def create_bottleneck_file_new(bottleneck_path, image_path, label_name,sess, inceptionV3Model):
  """Create a single bottleneck file."""
  print('Creating bottleneck at ' + bottleneck_path)

  if not tf.gfile.Exists(image_path):
    tf.logging.fatal('File does not exist %s', image_path)
  #image_data = tf.gfile.FastGFile(image_path, 'rb').read()
  image = Image.open(image_path)
  image_data = image.convert('RGB')
  try:
    bottleneck_values = inceptionV3Model.run_bottleneck_on_image(
        sess, image_data)
  except Exception as e:
	print e
	raise RuntimeError('Error during processing file %s' % image_path)

  bottleneck_string = ','.join(str(x) for x in bottleneck_values)
  with open(bottleneck_path, 'w') as bottleneck_file:
    bottleneck_file.write(bottleneck_string)

def create_bottleneck_file(bottleneck_path, image_lists, label_name, index,
                           image_dir, category, sess, inceptionV3Model):
  """Create a single bottleneck file."""
  print('Creating bottleneck at ' + bottleneck_path)
  image_path = get_image_path(image_lists, label_name, index,
                              image_dir, category)
  if not tf.gfile.Exists(image_path):
    tf.logging.fatal('File does not exist %s', image_path)
  #image_data = tf.gfile.FastGFile(image_path, 'rb').read()
  image = Image.open(image_path)
  image_data = image.convert('RGB')
  try:
    bottleneck_values = inceptionV3Model.run_bottleneck_on_image(
        sess, image_data)
  except Exception as e:
	print e
	raise RuntimeError('Error during processing file %s' % image_path)

  bottleneck_string = ','.join(str(x) for x in bottleneck_values)
  with open(bottleneck_path, 'w') as bottleneck_file:
    bottleneck_file.write(bottleneck_string)

def get_or_create_bottleneck_new(sess, image_path,image_label,bottleneck_dir,inceptionV3Model):
  """Retrieves or calculates bottleneck values for an image.
  If a cached version of the bottleneck data exists on-disk, return that,
  otherwise calculate the data and save it to disk for future use.
  Args:
    sess: The current active TensorFlow Session.
    image_lists:image path.
    bottleneck_dir: Folder string holding cached files of bottleneck values.
  Returns:
    Numpy array of values produced bget_bottleneck_pathy the bottleneck layer for the image.
  """
  sub_dir_path = os.path.join(bottleneck_dir, image_label)
  ensure_dir_exists(sub_dir_path)
  bottleneck_path = get_bottleneck_path_new(image_path, image_label, bottleneck_dir)

  if not os.path.exists(bottleneck_path):
    create_bottleneck_file_new(bottleneck_path, image_path, image_label, sess, inceptionV3Model)
  with open(bottleneck_path, 'r') as bottleneck_file:
    bottleneck_string = bottleneck_file.read()
  did_hit_error = False
  try:
    bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
  except ValueError:
    print('Invalid float found, recreating bottleneck')
    did_hit_error = True
  if did_hit_error:
    create_bottleneck_file_new(bottleneck_path, image_path, image_label, sess, inceptionV3Model)
    with open(bottleneck_path, 'r') as bottleneck_file:
      bottleneck_string = bottleneck_file.read()
    # Allow exceptions to propagate here, since they shouldn't happen after a
    # fresh creation
    bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
  return bottleneck_values


def get_or_create_bottleneck(sess, image_lists, label_name, index, image_dir,
                             category, bottleneck_dir,inceptionV3Model):
  """Retrieves or calculates bottleneck values for an image.
  If a cached version of the bottleneck data exists on-disk, return that,
  otherwise calculate the data and save it to disk for future use.
  Args:
    sess: The current active TensorFlow Session.
    image_lists: Dictionary of training images for each label.
    label_name: Label string we want to get an image for.
    index: Integer offset of the image we want. This will be modulo-ed by the
    available number of images for the label, so it can be arbitrarily large.
    image_dir: Root folder string  of the subfolders containing the training
    images.
    category: Name string of which  set to pull images from - training, testing,
    or validation.
    bottleneck_dir: Folder string holding cached files of bottleneck values.
    jpeg_data_tensor: The tensor to feed loaded jpeg data into.
    bottleneck_tensor: The output tensor for the bottleneck values.
  Returns:
    Numpy array of values produced bget_bottleneck_pathy the bottleneck layer for the image.
  """
  label_lists = image_lists[label_name]
  sub_dir = label_lists['dir']
  sub_dir_path = os.path.join(bottleneck_dir, sub_dir)
  ensure_dir_exists(sub_dir_path)
  bottleneck_path = get_bottleneck_path(image_lists, label_name, index,
                                        bottleneck_dir, category)
  if not os.path.exists(bottleneck_path):
    create_bottleneck_file(bottleneck_path, image_lists, label_name, index,
                           image_dir, category, sess, inceptionV3Model)
  with open(bottleneck_path, 'r') as bottleneck_file:
    bottleneck_string = bottleneck_file.read()
  did_hit_error = False
  try:
    bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
  except ValueError:
    print('Invalid float found, recreating bottleneck')
    did_hit_error = True
  if did_hit_error:
    create_bottleneck_file(bottleneck_path, image_lists, label_name, index,
                           image_dir, category, sess, inceptionV3Model)
    with open(bottleneck_path, 'r') as bottleneck_file:
      bottleneck_string = bottleneck_file.read()
    # Allow exceptions to propagate here, since they shouldn't happen after a
    # fresh creation
    bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
  return bottleneck_values

def get_random_cached_bottlenecks_new(sess, image_paths, image_labels, bottleneck_dir, inceptionV3Model):
  """Retrieves bottleneck values for cached images.
  If no distortions are being applied, this function can retrieve the cached
  bottleneck values directly from disk for images. It picks a random set of
  images from the specified category.
  Args:
    sess: Current TensorFlow Session.
    image_paths: List of image paths.
	image_labels: List of the image labels
    bottleneck_dir: Folder string holding cached files of bottleneck values.
  Returns:
    List of bottleneck arrays.
  """
  bottlenecks = []
  for i in range(len(image_paths)):
	  bottleneck = get_or_create_bottleneck_new(sess, image_paths[i], image_labels[i], bottleneck_dir, inceptionV3Model)
	  bottlenecks.append(bottleneck)
  return bottlenecks


def get_random_cached_bottlenecks(sess, image_lists, how_many, category,
                                  bottleneck_dir, image_dir, inceptionV3Model):
  """Retrieves bottleneck values for cached images.
  If no distortions are being applied, this function can retrieve the cached
  bottleneck values directly from disk for images. It picks a random set of
  images from the specified category.
  Args:
    sess: Current TensorFlow Session.
    image_lists: Dictionary of training images for each label.
    how_many: If positive, a random sample of this size will be chosen.
    If negative, all bottlenecks will be retrieved.
    category: Name string of which set to pull from - training, testing, or
    validation.
    bottleneck_dir: Folder string holding cached files of bottleneck values.
    image_dir: Root folder string of the subfolders containing the training
    images.
    jpeg_data_tensor: The layer to feed jpeg image data into.
    bottleneck_tensor: The bottleneck output layer of the CNN graph.
  Returns:
    List of bottleneck arrays, their corresponding ground truths, and the
    relevant filenames.
  """
  class_count = len(image_lists.keys())
  bottlenecks = []
  ground_truths = []
  filenames = []
  if how_many >= 0:
    # Retrieve a random sample of bottlenecks.
    for unused_i in range(how_many):
      label_index = random.randrange(class_count)
      label_name = list(image_lists.keys())[label_index]
      image_index = random.randrange(MAX_NUM_IMAGES_PER_CLASS + 1)
      image_name = get_image_path(image_lists, label_name, image_index,
                                  image_dir, category)
      bottleneck = get_or_create_bottleneck(sess, image_lists, label_name,
                                            image_index, image_dir, category,
                                            bottleneck_dir, inceptionV3Model)
      ground_truth = np.zeros(class_count, dtype=np.float32)
      ground_truth[label_index] = 1.0
      bottlenecks.append(bottleneck)
      ground_truths.append(ground_truth)
      filenames.append(image_name)
  else:
    # Retrieve all bottlenecks.
    for label_index, label_name in enumerate(image_lists.keys()):
      for image_index, image_name in enumerate(
          image_lists[label_name][category]):
        image_name = get_image_path(image_lists, label_name, image_index,
                                    image_dir, category)
        bottleneck = get_or_create_bottleneck(sess, image_lists, label_name,
                                              image_index, image_dir, category,
                                              bottleneck_dir,inceptionV3Model)
        ground_truth = np.zeros(class_count, dtype=np.float32)
        ground_truth[label_index] = 1.0
        bottlenecks.append(bottleneck)
        ground_truths.append(ground_truth)
        filenames.append(image_name)
  return bottlenecks, ground_truths, filenames

def get_random_distorted_bottlenecks(
    sess, image_paths, inceptionV3Model):
  """Retrieves bottleneck values for training images, after distortions.
  If we're training with distortions like crops, scales, or flips, we have to
  recalculate the full model for every image, and so we can't use cached
  bottleneck values. Instead we find random images for the requested category,
  run them through the distortion graph, and then the full graph to get the
  bottleneck results for each.
  Args:
    sess: Current TensorFlow Session.
    image_paths: List of image paths.
    input_jpeg_tensor: The input layer we feed the image data to.
    distorted_image: The output node of the distortion graph.
  Returns:
    List of bottleneck arrays and their corresponding ground truths.
  """
  bottlenecks = []
  for i in range(len(image_paths)):
      image = Image.open(image_paths[i])
      image_data = image.convert('RGB')
      distorted_image_data = inceptionV3Model.distort_image(sess,image_data)
      try:
        bottleneck = inceptionV3Model.run_bottleneck_on_image(sess, distorted_image_data)
        bottlenecks.append(bottleneck)
      except Exception as e:
        print e
        raise RuntimeError('Error during processing file %s' % image_paths[i])
  return bottlenecks

def should_distort_images(FLAGS):
    """Whether any distortions are enabled, from the input flags.
    Args:
    flip_left_right: Boolean whether to randomly mirror images horizontally.
    random_crop: Integer percentage setting the total margin used around the
    crop box.
    random_scale: Integer percentage of how much to vary the scale by.
    random_brightness: Integer range to randomly multiply the pixel values by.
    Returns:
    Boolean value indicating whether any distortions should be applied.
    """
    if(FLAGS.apply_distortions):
        return (FLAGS.flip_left_right or (FLAGS.random_crop != 0) or (FLAGS.random_scale != 0) or (FLAGS.random_brightness != 0))
    else:
        return False

def readDataset(FLAGS):
    # Look at the folder structure, and create lists of all the images.
    imageMap = create_image_lists(FLAGS.image_dir, FLAGS.testing_percentage,FLAGS.validation_percentage)
    save_labels(FLAGS,imageMap)
    return imageMap
