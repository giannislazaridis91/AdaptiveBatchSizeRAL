import numpy as np
import scipy
from tensorflow import keras
import tensorflow as tf

class Dataset:
    
    def __init__(self, number_of_state_data, train_dataset_length):

        # Initializes the Dataset object and initializes the attributes with given or empty values.
        self.train_data = np.array([[]])
        self.train_labels = np.array([[]])
        self.train_labels_one_hot_encoding = np.array([[]])
        self.test_data = np.array([[]])
        self.test_labels = np.array([[]])
        self.test_labels_one_hot_encoding = np.array([[]])
        self.state_data = np.array([[]])
        self.state_labels = np.array([[]])
        self.state_labels_one_hot_encoding = np.array([[]])
        self.number_of_state_data = number_of_state_data
        self.number_of_test_data = 10000
        self.number_of_classes = 0
        self.train_dataset_length = train_dataset_length
        self.regenerate()
        
    def regenerate(self):

        # The function for generating a dataset with new parameters.
        pass

    def _normalization(self):

        # Data normalization.
        self.train_data = self.train_data.astype('float32')
        self.test_data = self.test_data.astype('float32')
        self.state_data = self.state_data.astype('float32')
        self.train_data /= 255
        self.test_data /= 255
        self.state_data /= 255

    def _one_hot_encoding(self):

        # Convert class vectors to binary class matrices using one hot encoding.
        self.train_labels_one_hot_encoding = keras.utils.to_categorical(self.train_labels, num_classes = self.number_of_classes)
        self.test_labels_one_hot_encoding = keras.utils.to_categorical(self.test_labels, num_classes = self.number_of_classes)
        self.state_labels_one_hot_encoding  = keras.utils.to_categorical(self.state_labels, num_classes = self.number_of_classes)

    """
    def _compute_distances(self):

        # Create a new array with 50000 length
        features_array = np.zeros((len(self.train_data), 3072))

        # Loop through the training data and add the features to the new array
        for i in range(len(self.train_data)):
            # Get the features for the current image
            features = self.train_data[i]
            
            # Reshape the features to match the shape of the new array
            features = features.reshape(32, 32, 3).flatten()
            
            # Add the features to the new array
            features_array[i] = features

        # Computes the pairwise distances between all training images.
        self.distances = scipy.spatial.distance.pdist(features_array, metric='cosine')
        self.distances = scipy.spatial.distance.squareform(self.distances)
    """

    """
        self.distances = np.zeros((len(self.train_data),len(self.train_data)))
        for i in range(len(self.train_data)):
            i1 = self.train_data[i]
            for j in range(len(self.train_data)):
                i2 = self.train_data[j]
                self.distances[i][j] = np.sum((i1-i2)**2)
    """


class DatasetCIFAR10(Dataset):     

    def __init__(self, number_of_state_data, train_dataset_length):

        # Initialize a few attributes and the attributes of Dataset objects.
        Dataset.__init__(self, number_of_state_data, train_dataset_length)
    
    def regenerate(self):

        # Load CIFAR10 dataset
        (train_data, train_labels), (test_data, test_labels) = keras.datasets.cifar10.load_data()

        # Train data.
        new_data = []
        new_data_labels = []
        classes = [0,1,2,3,4,5,6,7,8,9]
        for data_class in classes:
            count = 0
            for i in range(len(train_labels)):
                if int(train_labels[i])==data_class and count < self.train_dataset_length/10:
                    count+=1
                    new_data.append(train_data[i])
                    new_data_labels.append(train_labels[i])
        for data_class in classes:
            count = 0
            for i in range(len(new_data_labels)):
                if int(new_data_labels[i])==data_class:
                        count+=1
        self.train_data = np.array(new_data)
        self.train_labels = np.array(new_data_labels)

        # Test data.
        new_data = []
        new_data_labels = []
        for data_class in classes:
            count = 0
            for i in range(len(test_labels)):
                if int(test_labels[i])==data_class and count < (self.number_of_test_data/10):
                    count+=1
                    new_data.append(test_data[i])
                    new_data_labels.append(test_labels[i])
        self.test_data = np.array(new_data)
        self.test_labels = np.array(new_data_labels)

        # State data.
        new_data = []
        new_data_labels = []
        for data_class in classes:
            count = 0
            for i in range(len(test_labels)):
                if int(test_labels[-i])==data_class and count < (self.number_of_state_data/10):
                    count+=1
                    new_data.append(test_data[-i])
                    new_data_labels.append(test_labels[-i])
        self.state_data = np.array(new_data)
        self.state_labels = np.array(new_data_labels)

        self.number_of_classes = len(np.unique(self.train_labels))
        self._normalization()
        self._one_hot_encoding()
        # self._compute_distances()