from PIL import Image, ImageOps
import os
import numpy as np

class ImagePreprocessor:
    def __init__(self, pixels=64, normalization=1, training_threshold=1, resize_method='square resize', color_mode='L'):
        '''
        Constructs the ImagePreprocessor object

        Args:
            pixels : Integer, number of pixels along each side of the processes image
            normalization : Integer, divisor that squeezes the color values
            training_threshold : Float, threshold that determines the proportion of data to be used for training
            resize_method : String, "square resize" squeezes the images into squares and "square crop" removes excess content to create a square image
            color_mode : String, see Pillow Modes documentation at https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes
        '''
        self.pixels = pixels
        self.normalization = normalization
        self.training_threshold = training_threshold
        self.resize_method = resize_method
        self.color_mode = color_mode
    
    def load_directory_contents(self, path):
        '''
        Returns the images in specified directory

        Args:
            path : String, path where the images are stored
        
        Returns:
            List : PIL image objects
        '''
        return [Image.open(f'{path}/{f}') for f in os.listdir(path)]

    def prepare_image(self, image):
        '''
        Performs the preprocess method on the input image and/or resizes it

        Args:
            image : PIL Image, image to be preprocessed
        
        Returns:
            PIL Image : Preprocessed image
        '''
        if self.resize_method == 'square crop':
            width, height = image.size
            h_cut = v_cut = 0

            if width > height:
                h_cut = (width - height)/2
            elif width < height:
                v_cut = (height - width)/2

            image = ImageOps.crop(image, (h_cut, v_cut, h_cut, v_cut))
        image = image.resize((self.pixels, self.pixels), Image.BILINEAR).convert(self.color_mode)
        return image

    def directory_to_array(self, path):
        '''
        Returns a NumPy array containing image data from specified directory

        Args:
            path : String, path of directory
        
        Returns:
            NumPy array : array representations of images
        '''
        cropped = [self.prepare_image(image) for image in self.load_directory_contents(path)]
        arr = [np.array(image) for image in cropped]
        return np.array(arr)/self.normalization
    
    def count_channels(self, image):
        '''
        Counts the number of color channels in the given image

        Args:
            image : NumPy array, array representation of image
        
        Returns:
            Integer : number of color channels in the image
        '''
        if isinstance(image[0][0], np.ndarray):
            return len(image[0][0])
        return 1

    def preprocess_dirs(self, paths, labels, partition=False):
        '''
        Returns a dict containing preprocessed images and their respective labels.

        Args:
            paths : List, directories to be preprocessed
            labels : List, labels of each directory
            partition : Boolean, dictates if the images should be partitioned into training and testing data
        
        Returns:
            Dict : Preprocessed and partitioned data
        '''
        train_features = []
        train_labels = []
        test_features = []
        test_labels = []

        for label_index, path in enumerate(paths):
            cropped = [self.prepare_image(image) for image in self.load_directory_contents(path)]
            np.random.shuffle(cropped)
            threshold = int(len(cropped)*self.training_threshold)
            dir_images = [np.array(image) for image in cropped]

            for i, image in enumerate(dir_images):
                if i <= threshold or not partition:
                    train_features.append(image)
                    train_labels.append(labels[label_index])
                else:
                    test_features.append(image)
                    test_labels.append(labels[label_index])
        
        channels = self.count_channels(train_features[0])

        train_features = np.array(train_features).reshape(-1, self.pixels, self.pixels, channels)/self.normalization
        train_labels = np.array(train_labels)
        test_features = np.array(test_features).reshape(-1, self.pixels, self.pixels, channels)/self.normalization
        test_labels = np.array(test_labels)
        
        package = {
            'TRAIN_IMAGES' : train_features,
            'TRAIN_LABELS' : train_labels,
            'TEST_IMAGES' : test_features,
            'TEST_LABELS' : test_labels
        }

        return package

    def file_to_array(self, path):
        '''
        Returns a NumPy array representation of image specified by path

        Args:
            path : String, path of image to be converted into an array

        Returns:
            NumPy array : array representation of preprocessed input image
        '''
        image = np.array(self.prepare_image(Image.open(path)))
        image = image.reshape(-1, self.pixels, self.pixels, self.count_channels(image))/self.normalization
        return image