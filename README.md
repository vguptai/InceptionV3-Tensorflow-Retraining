# InceptionV3-Tensorflow-Retraining

This repository is inspired by the [Image Retraining](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/image_retraining) example. Some code has been trimmed off and refactored as per the use case.

The code adds untrained (depending on the number of classes) fully connected layers to the loaded InceptionV3 graph. The loaded InceptionV3 model is NOT FINE TUNED. It is only used to generate the features for the images. However, the fully connected layers are trained.

Run the following script to train the network.
```python
python trainer.py
```

Run the following script for testing the network.
```python
python tester.py
```

<b> Dataset </b>

The code expects the data to be organized in a parent folder with images of different classes in different folders.

For Example,
cifar10Dataset - Parent Directory
cifar10Dataset/cat
cifar10Dataset/dog

<b> ToDo </b>

1) Add support for handling png images - Done
2) Add batching support - Done
3) Support for saving and reloading model for inference - Done
4) Support for adding distortions to images while training - Done
