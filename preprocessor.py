import tensorflow as tf
import tarfile
from osUtils import *
import random
from constants import *
from six.moves import urllib
import sys
import numpy as np
from PIL import Image

def setupSummariesDirectory(summariesDir):
	# Setup the directory we'll write summaries to for TensorBoard
	if tf.gfile.Exists(summariesDir):
		tf.gfile.DeleteRecursively(summariesDir)
	tf.gfile.MakeDirs(summariesDir)

def download_and_extract_inception_model(modelDir):
  """Download and extract model tar file.
  If the pretrained model we're using doesn't already exist, this function
  downloads it from the TensorFlow.org website and unpacks it into a directory.
  """
  destDirectory = modelDir
  if not os.path.exists(destDirectory):
    os.makedirs(destDirectory)
  filename = INCEPTION_MODEL_URL.split('/')[-1]
  filepath = os.path.join(destDirectory, filename)
  if not os.path.exists(filepath):

    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' %
                       (filename,
                        float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()

    filepath, _ = urllib.request.urlretrieve(INCEPTION_MODEL_URL,
                                             filepath,
                                             _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')

  tarfile.open(filepath, 'r:gz').extractall(destDirectory)

def setup(FLAGS):
	setupSummariesDirectory(FLAGS.summaries_dir)
	download_and_extract_inception_model(FLAGS.imagenet_inception_model_dir)
