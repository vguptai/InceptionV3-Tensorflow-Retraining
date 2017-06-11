import os.path
import numpy as np
from random import shuffle

class DatasetBatcher:

    dataset_map = None
    training_images = []
    testing_images = []
    validation_images = []
    training_labels_one_hot_vector = None
    training_class_label = []
    testing_labels_one_hot_vector = None
    testing_class_label = []
    validation_labels_one_hot_vector = None
    validation_class_label = []
    training_data_offset = 0
    testing_data_offset = 0
    validation_data_offset = 0

    def __init__(self,dataset_map_,image_dir):
        print "Initializing dataset batcher..."
        self.dataset_map = dataset_map_
        self._flatten_map(image_dir)


    def _stack_vertically(self,target,source):
        if target==None:
            target = source
        else:
            target = np.vstack((target,source))
        return target

    def _flatten_map(self,image_dir):
        num_classes = len(self.dataset_map.keys())
        print "Number of classes:"+str(num_classes)
        labels_array = []
        for class_index, class_name in enumerate(self.dataset_map.keys()):
            self.training_images.extend(self._create_fully_qualified_path_for_images(image_dir,self.dataset_map[class_name]['training'],class_name))
            labels_array = self._create_one_hot_vector_matrix(len(self.dataset_map[class_name]['training']),class_index,num_classes)
            self.training_labels_one_hot_vector = self._stack_vertically(self.training_labels_one_hot_vector,labels_array)
            self.training_class_label.extend(self._duplicate_class_label(len(self.dataset_map[class_name]['training']),class_name))

            self.testing_images.extend(self._create_fully_qualified_path_for_images(image_dir,self.dataset_map[class_name]['testing'],class_name))
            labels_array = self._create_one_hot_vector_matrix(len(self.dataset_map[class_name]['testing']),class_index,num_classes)
            self.testing_labels_one_hot_vector = self._stack_vertically(self.testing_labels_one_hot_vector,labels_array)
            self.testing_class_label.extend(self._duplicate_class_label(len(self.dataset_map[class_name]['testing']),class_name))

            self.validation_images.extend(self._create_fully_qualified_path_for_images(image_dir,self.dataset_map[class_name]['validation'],class_name))
            labels_array = self._create_one_hot_vector_matrix(len(self.dataset_map[class_name]['validation']),class_index,num_classes)
            self.validation_labels_one_hot_vector = self._stack_vertically(self.validation_labels_one_hot_vector,labels_array)
            self.validation_class_label.extend(self._duplicate_class_label(len(self.dataset_map[class_name]['validation']),class_name))

        self.training_images,self.training_labels_one_hot_vector,self.training_class_label = self._shuffle(self.training_images,self.training_labels_one_hot_vector,self.training_class_label)
        self.testing_images,self.testing_labels_one_hot_vector,self.testing_class_label = self._shuffle(self.testing_images,self.testing_labels_one_hot_vector,self.testing_class_label)
        self.validation_images,self.validation_labels_one_hot_vector,self.validation_class_label = self._shuffle(self.validation_images,self.validation_labels_one_hot_vector,self.validation_class_label)

        print "Number of training labels:"+str(len(self.training_class_label))
        print "Number of testing labels:"+str(len(self.testing_class_label))
        print "Number of validation labels:"+str(len(self.validation_class_label))

    """
    Shuffles the dataset.
    file_list: list of the file paths
    file_label_one_hot_vectors: numpy array containing one hot encoded vectors
    file_labels: list containing names of the classe of the images
    """
    def _shuffle(self,file_list,file_label_one_hot_vectors,file_labels):
        file_list_shuff = []
        file_label_one_hot_vectors_shuff = None
        file_labels_shuff = []
        index_shuf = range(len(file_list))
        shuffle(index_shuf)
        for i in index_shuf:
            file_list_shuff.append(file_list[i])
            file_labels_shuff.append(file_labels[i])
            file_label_one_hot_vectors_shuff = self._stack_vertically(file_label_one_hot_vectors_shuff,file_label_one_hot_vectors[i,:])
        return file_list_shuff,file_label_one_hot_vectors_shuff,file_labels_shuff

    def _create_fully_qualified_path_for_images(self,image_dir,image_base_name_list,image_class_name):
        return [os.path.join(image_dir, image_class_name, image_base_name) for image_base_name in image_base_name_list]

    def _create_one_hot_vector_matrix(self,num_rows,class_index,num_classes):
        one_hot_array = np.zeros((num_rows,num_classes), dtype=np.float32)
        one_hot_array[:,class_index] = 1.0
        return one_hot_array

    def _duplicate_class_label(self,length_of_list,class_name):
        dup_labels = []
        for l in range(length_of_list):
            dup_labels.append(class_name)
        return dup_labels

    """
    Creates a new batch from the dataset
    images_list: list containing all the images
    image_labels_matrix: numpy matrix of one hot encoded vectors
    image_labels: list containing names of the classe of the images
    """
    def _create_next_batch_(self,batch_size,offset,images_list,image_labels_matrix,image_labels):
        if(offset>=len(images_list)):
            return None,None,None
        elif ((offset+batch_size)>len(images_list)):
            min_index = offset
            max_index = len(images_list)
        else:
            min_index = offset
            max_index = offset+batch_size

        image_paths = images_list[min_index:max_index]
        labels_matrix = image_labels_matrix[min_index:max_index,:]
        labels =image_labels[min_index:max_index]
        return image_paths,labels_matrix,labels

    def reset_training_offset(self,shuffle=False):
        self.training_data_offset = 0
        if shuffle:
            self.training_images,self.training_labels_one_hot_vector,self.training_class_label = self._shuffle(self.training_images,self.training_labels_one_hot_vector,self.training_class_label)

    def reset_testing_offset(self):
        self.testing_data_offset = 0

    def reset_validation_offset(self):
        self.validation_data_offset = 0

    def get_next_training_batch(self,batch_size):
        return self._get_next_batch('training',batch_size)

    def get_next_testing_batch(self,batch_size):
        return self._get_next_batch('testing',batch_size)

    def get_next_validation_batch(self,batch_size):
        return self._get_next_batch('validation',batch_size)

    """
    category - training,testing,validation
    batch_size - number of samples to return
    """
    def _get_next_batch(self,category,batch_size):
        if category=='training':
            image_paths,labels_matrix,labels = self._create_next_batch_(batch_size,self.training_data_offset,self.training_images,self.training_labels_one_hot_vector,self.training_class_label)
            self.training_data_offset = self.training_data_offset + batch_size
        elif category=='testing':
            image_paths,labels_matrix,labels = self._create_next_batch_(batch_size,self.testing_data_offset,self.testing_images,self.testing_labels_one_hot_vector,self.testing_class_label)
            self.testing_data_offset = self.testing_data_offset + batch_size
        elif category=='validation':
            image_paths,labels_matrix,labels = self._create_next_batch_(batch_size,self.validation_data_offset,self.validation_images,self.validation_labels_one_hot_vector,self.validation_class_label)
            self.validation_data_offset = self.validation_data_offset+batch_size
        return image_paths,labels_matrix,labels

    def number_of_training_batches(self,batch_size):
        return int(len(self.training_images)/batch_size)
